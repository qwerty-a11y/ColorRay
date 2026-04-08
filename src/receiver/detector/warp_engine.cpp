#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip> 
#include <omp.h>   

using namespace cv;
using namespace std;

/*==============================================================================
                                [极限性能模块]
==============================================================================*/
void FastIntegralThreshold(const Mat& src, Mat& dst, int blockSize, int C) {
    dst.create(src.size(), CV_8UC1);
    Mat sum;
    integral(src, sum, CV_32S);

    int half_sz = blockSize / 2;
    int rows = src.rows;
    int cols = src.cols;

    #pragma omp parallel for
    for (int y = 0; y < rows; ++y) {
        const uchar* src_ptr = src.ptr<uchar>(y);
        uchar* dst_ptr = dst.ptr<uchar>(y);

        int y1 = std::max(0, y - half_sz);
        int y2 = std::min(rows, y + half_sz + 1);
        const int* sum_y1 = sum.ptr<int>(y1);
        const int* sum_y2 = sum.ptr<int>(y2);

        for (int x = 0; x < cols; ++x) {
            int x1 = std::max(0, x - half_sz);
            int x2 = std::min(cols, x + half_sz + 1);

            int count = (y2 - y1) * (x2 - x1);
            int s = sum_y2[x2] - sum_y2[x1] - sum_y1[x2] + sum_y1[x1];
            if (src_ptr[x] * count <= s - C * count) dst_ptr[x] = 255;
            else dst_ptr[x] = 0;
        }
    }
}

/*==============================================================================
                                [全局识别模块]
==============================================================================*/
vector<Point3f> FindMarkerCenters(const Mat& input, int ch) {
    Mat gray;
    if (ch == 3) cvtColor(input, gray, COLOR_BGR2GRAY);
    else gray = input.clone();

    Mat binary;
    FastIntegralThreshold(gray, binary, 101, 15);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat closed_binary;
    morphologyEx(binary, closed_binary, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<Point3f> centers;
    for (size_t i = 0; i < contours.size(); i++) {
        int kid_idx = hierarchy[i][2];
        int cnt = 0;
        while (kid_idx != -1) {
            cnt++;
            kid_idx = hierarchy[kid_idx][2];
            if (cnt >= 2) break;
        }
        if (cnt >= 2) {
            Moments mu = moments(contours[i], false);
            if (mu.m00 >= 500) {
                centers.push_back(Point3f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00), (float)mu.m00));
            }
        }
    }

    vector<Point3f> merged;
    for (const auto& pt : centers) {
        bool is_new = true;
        for (auto& m : merged) {
            if (norm(Point2f(pt.x, pt.y) - Point2f(m.x, m.y)) < 150.0) {
                is_new = false;
                if (pt.z > m.z) m = pt;
                break;
            }
        }
        if (is_new) merged.push_back(pt);
    }
    return merged;
}

/*==============================================================================
                                [局部追踪模块] 
==============================================================================*/
Point3f FindSingleMarkerInROI(const Mat& roi_img, int ch) {
    Mat gray;
    if (ch == 3) cvtColor(roi_img, gray, COLOR_BGR2GRAY);
    else gray = roi_img.clone();

    Mat binary;
    FastIntegralThreshold(gray, binary, 51, 17);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));
    Mat closed_binary;
    morphologyEx(binary, closed_binary, MORPH_OPEN, kernel);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    Point3f best_center(-1, -1, -1);
    float max_area = 0;

    for (size_t i = 0; i < contours.size(); i++) {
        int kid_idx = hierarchy[i][2];
        int cnt = 0;
        while (kid_idx != -1) {
            cnt++;
            kid_idx = hierarchy[kid_idx][2];
            if (cnt >= 2) break;
        }
        if (cnt >= 2) {
            Moments mu = moments(contours[i], false);
            if (mu.m00 > 50 && mu.m00 > max_area) { 
                max_area = (float)mu.m00;
                best_center = Point3f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00), (float)mu.m00);
            }
        }
    }
    return best_center;
}

/*==============================================================================
                            [解码模块：自动适配参数]
==============================================================================*/
void DecodePayload(const Mat& warped_img, unsigned char* decoded_data, int grid_size, int quiet_zone) {
    if (decoded_data == nullptr) return;

    int padded_size = grid_size + 2 * quiet_zone;
    int target_dim = padded_size * 11;

    Mat img_resized;
    resize(warped_img, img_resized, Size(target_dim, target_dim), 0, 0, INTER_LANCZOS4);
    cvtColor(img_resized, img_resized, COLOR_BGR2RGB);

    Scalar current_means = mean(img_resized);
    float gain_r = std::min(std::max(128.0f / (float)(current_means[0] + 1e-6), 0.1f), 5.0f);
    float gain_g = std::min(std::max(128.0f / (float)(current_means[1] + 1e-6), 0.1f), 5.0f);
    float gain_b = std::min(std::max(128.0f / (float)(current_means[2] + 1e-6), 0.1f), 5.0f);

    #pragma omp parallel for
    for (int y = 0; y < img_resized.rows; ++y) {
        Vec3b* ptr = img_resized.ptr<Vec3b>(y);
        for (int x = 0; x < img_resized.cols; ++x) {
            ptr[x][0] = saturate_cast<uchar>(ptr[x][0] * gain_r);
            ptr[x][1] = saturate_cast<uchar>(ptr[x][1] * gain_g);
            ptr[x][2] = saturate_cast<uchar>(ptr[x][2] * gain_b);
        }
    }

    float kernel[11][11];
    float sum = 0;
    for (int i = 0; i < 11; ++i) {
        float y = i - 5.0f;
        for (int j = 0; j < 11; ++j) {
            float x = j - 5.0f;
            kernel[i][j] = std::exp(-0.5f * (x * x + y * y) / (0.5f * 0.5f));
            sum += kernel[i][j];
        }
    }
    for (int i = 0; i < 11; ++i) for (int j = 0; j < 11; ++j) kernel[i][j] /= sum;

    Mat final_array(padded_size, padded_size, CV_8UC3, Scalar(255, 255, 255));
    
    #pragma omp parallel for
    for (int by = 0; by < padded_size; ++by) {
        for (int bx = 0; bx < padded_size; ++bx) {
            float r = 0, g = 0, b = 0;
            for (int ky = 0; ky < 11; ++ky) {
                const Vec3b* row_ptr = img_resized.ptr<Vec3b>(by * 11 + ky);
                for (int kx = 0; kx < 11; ++kx) {
                    float w = kernel[ky][kx];
                    const Vec3b& px = row_ptr[bx * 11 + kx];
                    r += px[0] * w; g += px[1] * w; b += px[2] * w;
                }
            }
            Vec3b& dst = final_array.at<Vec3b>(by, bx);
            dst[0] = r < 128.0f ? 0 : 255;
            dst[1] = g < 128.0f ? 0 : 255;
            dst[2] = b < 128.0f ? 0 : 255;
        }
    }
    std::memcpy(decoded_data, final_array.data, padded_size * padded_size * 3);
}

/*==============================================================================
                                    [接口函数]
==============================================================================*/
extern "C" {

static vector<Point2f> prev_corners; 
static bool use_tracking = false;
static double total_time_ms = 0;
static long long frame_counter = 0;

__declspec(dllexport) bool ExtractQRCode(
    unsigned char* in_data, int width, int height, int channels,
    unsigned char* out_data, int out_width, int out_height,
    int grid_size, int quiet_zone, int large_finder, 
    unsigned char* decoded_data)
{
    auto start = std::chrono::high_resolution_clock::now();

    Mat img(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, in_data);
    Mat out_img(out_height, out_width, CV_8UC3, out_data);
    
    vector<Point2f> current_corners;
    bool tracking_success = false;

    if (use_tracking && prev_corners.size() == 4) {
        tracking_success = true;
        int roi_size = 200; 
        current_corners.resize(4); 
        bool track_failed_flag = false;

        #pragma omp parallel for
        for (int i = 0; i < 4; i++) {
            if (track_failed_flag) continue; 
            Point2f pt = prev_corners[i];
            int rx = std::max(0, (int)pt.x - roi_size / 2);
            int ry = std::max(0, (int)pt.y - roi_size / 2);
            int rw = std::min(width - rx, roi_size);
            int rh = std::min(height - ry, roi_size);
            Rect roi_rect(rx, ry, rw, rh);
            Mat roi_img = img(roi_rect);
            Point3f local_center = FindSingleMarkerInROI(roi_img, channels);
            
            if (local_center.z > 0) current_corners[i] = Point2f(local_center.x + rx, local_center.y + ry);
            else track_failed_flag = true;
        }
        if (track_failed_flag) { tracking_success = false; current_corners.clear(); }
    }

    if (!tracking_success) {
        float scale = 1100.0f / (float)max(width, height);
        Mat small_img;
        resize(img, small_img, Size(), scale, scale, INTER_AREA);
        vector<Point3f> centers;
        vector<Point3f> small_centers = FindMarkerCenters(small_img, channels);
        for (auto pt : small_centers) centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));

        if (centers.size() >= 4) {
            Point2f approx_center(0, 0);
            for (auto p : centers) approx_center += Point2f(p.x, p.y);
            approx_center.x /= (float)centers.size(); approx_center.y /= (float)centers.size();
            if (centers.size() > 4) {
                sort(centers.begin(), centers.end(), [&approx_center](Point3f a, Point3f b) {
                    return norm(Point2f(a.x, a.y) - approx_center) > norm(Point2f(b.x, b.y) - approx_center);
                });
                centers.resize(4);
            }
            Point2f exact_center(0, 0);
            for (auto p : centers) exact_center += Point2f(p.x, p.y);
            exact_center.x /= 4.0f; exact_center.y /= 4.0f;
            int br_idx = -1; float min_ratio = 1e9;
            for (int i = 0; i < 4; i++) {
                float dist = (float)norm(Point2f(centers[i].x, centers[i].y) - exact_center);
                float ratio = centers[i].z / (dist * dist);
                if (ratio < min_ratio) { min_ratio = ratio; br_idx = i; }
            }
            Point2f br_point = Point2f(centers[br_idx].x, centers[br_idx].y);
            vector<Point2f> sorted_pts;
            for (auto p : centers) sorted_pts.push_back(Point2f(p.x, p.y));
            sort(sorted_pts.begin(), sorted_pts.end(), [&exact_center](Point2f a, Point2f b) {
                return atan2(a.y - exact_center.y, a.x - exact_center.x) < atan2(b.y - exact_center.y, b.x - exact_center.x);
            });
            int s_br_idx = 0;
            for (int i = 0; i < 4; i++) if (norm(sorted_pts[i] - br_point) < 1.0f) { s_br_idx = i; break; }
            current_corners.push_back(sorted_pts[(s_br_idx + 2) % 4]); 
            current_corners.push_back(sorted_pts[(s_br_idx + 3) % 4]); 
            current_corners.push_back(sorted_pts[s_br_idx]);           
            current_corners.push_back(sorted_pts[(s_br_idx + 1) % 4]); 
        }
    }

    if (current_corners.size() == 4) {
        if (use_tracking) {
            float alpha = 0.95f; float deadzone_radius = 0.5f; 
            for (int i = 0; i < 4; i++) {
                float dist = (float)norm(current_corners[i] - prev_corners[i]);
                if (dist < deadzone_radius) current_corners[i] = prev_corners[i];
                else current_corners[i] = current_corners[i] * alpha + prev_corners[i] * (1.0f - alpha);
            }
        }
        prev_corners = current_corners;
        use_tracking = true;
        Point2f tl = current_corners[0], tr = current_corners[1], br = current_corners[2], bl = current_corners[3];
/*==============================================================================
                                [自适应数学矫正模块]
==============================================================================*/
        float logic_total_width = (float)grid_size + 2.0f * (float)quiet_zone;
        float center_offset = (float)quiet_zone + (float)large_finder / 2.0f ;
        float r_min = center_offset / logic_total_width;
        float r_max = (logic_total_width - center_offset) / logic_total_width;

        vector<Point2f> src = { tl, tr, br, bl };
        vector<Point2f> dst = { 
            Point2f(r_min * out_width, r_min * out_height), 
            Point2f(r_max * out_width, r_min * out_height), 
            Point2f(r_max * out_width, r_max * out_height), 
            Point2f(r_min * out_width, r_max * out_height) 
        };
        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(img, out_img, M, Size(out_width, out_height), INTER_NEAREST);
        if (decoded_data != nullptr) DecodePayload(out_img, decoded_data, grid_size, quiet_zone);
    } else { use_tracking = false; return false; }

/*==============================================================================
                                [性能监控模块]
==============================================================================*/
    auto end = std::chrono::high_resolution_clock::now();
    double current_duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    total_time_ms += current_duration;
    frame_counter++;
    double average_duration = total_time_ms / (double)frame_counter;

    std::cout << ">>> [CPU Warp Engine] Mode: " << (tracking_success ? "Tracking" : "Global") 
              << " | Current: " << std::fixed << std::setprecision(1) << current_duration << " ms"
              << " | Average: " << std::fixed << std::setprecision(2) << average_duration << " ms" 
              << " (" << frame_counter << " frames)\r";

    return true;
}
}