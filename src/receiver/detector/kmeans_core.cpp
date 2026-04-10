#include <opencv2/opencv.hpp>
#include <omp.h>
#include <vector>
#include <random>
#include <limits>
#include <cmath>
#include <cstdint>
#include <algorithm>

extern "C" {

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// ----------------------------------------------------------------------------
// 内部纯净 K-Means++ 引擎 (严格对齐 sklearn)
// ----------------------------------------------------------------------------
struct KMeansResult {
    std::vector<int> labels;
    std::vector<cv::Vec3f> centers;
    float inertia;
};

KMeansResult CustomKMeansPlusPlus(const std::vector<cv::Vec3f>& data, int K, int n_init, int max_iter, int random_state) {
    KMeansResult best_result;
    best_result.inertia = std::numeric_limits<float>::max();
    std::mt19937 rng(random_state); 
    int N = data.size();

    for (int init_run = 0; init_run < n_init; ++init_run) {
        std::vector<cv::Vec3f> centers;
        centers.reserve(K);
        
        std::uniform_int_distribution<int> dist_first(0, N - 1);
        centers.push_back(data[dist_first(rng)]);

        std::vector<float> min_dist_sq(N, std::numeric_limits<float>::max());
        for (int k = 1; k < K; ++k) {
            float sum_dist_sq = 0.0f;
            #pragma omp parallel for reduction(+:sum_dist_sq)
            for (int i = 0; i < N; ++i) {
                float d0 = data[i][0] - centers.back()[0];
                float d1 = data[i][1] - centers.back()[1];
                float d2 = data[i][2] - centers.back()[2];
                float dist = d0 * d0 + d1 * d1 + d2 * d2;
                if (dist < min_dist_sq[i]) min_dist_sq[i] = dist;
                sum_dist_sq += min_dist_sq[i];
            }

            std::uniform_real_distribution<float> dist_prob(0.0f, sum_dist_sq);
            float target = dist_prob(rng);
            float cumulative = 0.0f;
            int next_center_idx = N - 1;
            for (int i = 0; i < N; ++i) {
                cumulative += min_dist_sq[i];
                if (cumulative >= target) {
                    next_center_idx = i;
                    break;
                }
            }
            centers.push_back(data[next_center_idx]);
        }

        std::vector<int> labels(N, 0);
        float current_inertia = 0.0f;
        
        for (int iter = 0; iter < max_iter; ++iter) {
            current_inertia = 0.0f;
            std::vector<cv::Vec3f> new_centers(K, cv::Vec3f(0, 0, 0));
            std::vector<int> counts(K, 0);

            #pragma omp parallel
            {
                float local_inertia = 0.0f;
                std::vector<cv::Vec3f> local_new_centers(K, cv::Vec3f(0, 0, 0));
                std::vector<int> local_counts(K, 0);

                #pragma omp for schedule(static) nowait
                for (int i = 0; i < N; ++i) {
                    float best_dist = std::numeric_limits<float>::max();
                    int best_k = 0;
                    for (int k = 0; k < K; ++k) {
                        float d0 = data[i][0] - centers[k][0];
                        float d1 = data[i][1] - centers[k][1];
                        float d2 = data[i][2] - centers[k][2];
                        float dist = d0 * d0 + d1 * d1 + d2 * d2;
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                    labels[i] = best_k;
                    local_inertia += best_dist;
                    local_new_centers[best_k] += data[i];
                    local_counts[best_k]++;
                }

                #pragma omp critical
                {
                    current_inertia += local_inertia;
                    for (int k = 0; k < K; ++k) {
                        new_centers[k] += local_new_centers[k];
                        counts[k] += local_counts[k];
                    }
                }
            }

            bool changed = false;
            for (int k = 0; k < K; ++k) {
                if (counts[k] > 0) {
                    cv::Vec3f avg = new_centers[k] / (float)counts[k];
                    if (cv::norm(avg - centers[k]) > 1e-4) {
                        centers[k] = avg;
                        changed = true;
                    }
                }
            }
            if (!changed) break;
        }

        if (current_inertia < best_result.inertia) {
            best_result.inertia = current_inertia;
            best_result.centers = centers;
            best_result.labels = labels;
        }
    }
    return best_result;
}

// ----------------------------------------------------------------------------
// 全流程 C++ 接管：方差预处理 -> 采样 -> 聚类 -> 重构
// ----------------------------------------------------------------------------
EXPORT_API bool ProcessColorEngine(
    const uint8_t* in_data, int width, int height, 
    uint8_t* out_data, int out_scale, int grid_size)
{
    if (!in_data || !out_data) return false;

    // Intel 平台火力全开：依托 OpenCV 底层原生的 Intel IPP 硬件加速，并拉满 OpenMP 线程
    omp_set_num_threads(omp_get_max_threads());

    cv::Mat img_bgr(height, width, CV_8UC3, (void*)in_data);
    
    // 1. 预处理 (OpenCV 加速, 1:1 复刻 Python 的 E_x2 - E_x**2)
    // 这里 OpenCV 在 Intel CPU 上会自动调用 IPP 库实现最极致的 AVX 向量化运算
    cv::Mat img_float, img_sq;
    img_bgr.convertTo(img_float, CV_32FC3);
    cv::multiply(img_float, img_float, img_sq);

    cv::Mat E_x, E_x2;
    cv::blur(img_float, E_x, cv::Size(3, 3));
    cv::blur(img_sq, E_x2, cv::Size(3, 3));

    cv::Mat E_x_sq, var_img_3c, var_img;
    cv::multiply(E_x, E_x, E_x_sq);
    cv::subtract(E_x2, E_x_sq, var_img_3c);
    cv::transform(var_img_3c, var_img, cv::Matx13f(1.0f, 1.0f, 1.0f)); 

    // 2. 局部方差游走采样 (完全移入 C++)
    float cell_w = (float)width / grid_size;
    float cell_h = (float)height / grid_size;
    int search_r = std::max(1, (int)(std::min(cell_w, cell_h) * 0.35f));

    int num_samples = grid_size * grid_size;
    std::vector<cv::Vec3f> sampled_colors(num_samples);

    #pragma omp parallel for collapse(2) schedule(guided)
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            int cx = (int)(col * cell_w + cell_w / 2.0f);
            int cy = (int)(row * cell_h + cell_h / 2.0f);
            
            cx = std::max(search_r, std::min(cx, width - search_r - 1));
            cy = std::max(search_r, std::min(cy, height - search_r - 1));

            float min_var = std::numeric_limits<float>::max();
            int best_cx = cx, best_cy = cy;

            for (int dy = -search_r; dy <= search_r; ++dy) {
                for (int dx = -search_r; dx <= search_r; ++dx) {
                    float var = var_img.at<float>(cy + dy, cx + dx);
                    if (var < min_var) {
                        min_var = var;
                        best_cx = cx + dx;
                        best_cy = cy + dy;
                    }
                }
            }
            sampled_colors[row * grid_size + col] = E_x.at<cv::Vec3f>(best_cy, best_cx);
        }
    }

    // 3. 调用手写 K-Means++ (参数：n_clusters=8, n_init=3, max_iter=300, random_state=42)
    KMeansResult km_res = CustomKMeansPlusPlus(sampled_colors, 8, 3, 300, 42);

    // 4. 标准化中心点 ( > 127 ? 255 : 0 )
    std::vector<cv::Vec3b> standard_colors(8);
    for (int i = 0; i < 8; ++i) {
        standard_colors[i] = cv::Vec3b(
            km_res.centers[i][0] > 127.0f ? 255 : 0,
            km_res.centers[i][1] > 127.0f ? 255 : 0,
            km_res.centers[i][2] > 127.0f ? 255 : 0
        );
    }

    // 5. Kronecker 极速重构图像回写
    int out_size = grid_size * out_scale;
    cv::Mat out_img(out_size, out_size, CV_8UC3, (void*)out_data);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < grid_size; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            int label = km_res.labels[row * grid_size + col];
            cv::Vec3b color = standard_colors[label];
            
            for (int dy = 0; dy < out_scale; ++dy) {
                for (int dx = 0; dx < out_scale; ++dx) {
                    out_img.at<cv::Vec3b>(row * out_scale + dy, col * out_scale + dx) = color;
                }
            }
        }
    }

    return true;
}

} // extern "C"