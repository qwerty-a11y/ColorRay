#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip> 

using namespace cv;
using namespace std;
/*                 [性能大升级！我想到了一个新的ai想不到的算法]
我们做图像处理我们实际上是分成两张图，一张找定位块，一张保留细节做拉伸，那既然如此对与我的程序而言只有
定位块的数据是有用的，其他数据都是无效数据甚至干扰数据，那我们要怎么优化呢？想想icpc的主席树动态开点对
于没用的数据就不要动了，我们只修改局部，还有很多树上操作也是第一遍全局查询再局部修改局部查询然后修正误
差，欸那既然如此，你想想，60fps的视频，是不是前后两帧定位块的位置很接近，那我们直接在原来的位置上做个
范围查询不就完了嘛？中间的和外部的环境干扰数据就统统屏蔽掉了，本身就自带抗干扰效果，还有性能优化，甚至
是视频矫正准确度大升级，一口气优化掉多个参数，这么天才的算法，ai是绝对不会主动想到的，只有我这个绝世大
级大天才能想到，super big cup算法!!!!你要是看了我的代码，看了这段讲话，就权当你学会了这算法思想了吧
                                                                    ————2024.3.22.linjunhao
*/
/*==============================================================================
                                [全局识别模块]
                功能: 全局盲扫，在丢失目标或第一帧时使用
==============================================================================*/
vector<Point3f> FindMarkerCenters(const Mat& input, int ch) {
    Mat gray;
    if (ch == 3) {
        Mat hsv, BinaryMask;
        cvtColor(input, hsv, COLOR_BGR2HSV);
        cvtColor(input, gray, COLOR_BGR2GRAY);
        vector<Mat> hsv_ch;
        split(hsv, hsv_ch);
        threshold(hsv_ch[1], BinaryMask, 180, 255, THRESH_BINARY);
        gray.setTo(255, BinaryMask);
    } else {
        gray = input.clone();
    }

    Mat binary;
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 15);
    
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
            if (mu.m00 >= 700) {
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
                在 4K 原图切出的小框里快速精准寻找唯一的定位块
==============================================================================*/
Point3f FindSingleMarkerInROI(const Mat& roi_img, int ch) {
    Mat gray;
    if (ch == 3) {
        Mat hsv, BinaryMask;
        cvtColor(roi_img, hsv, COLOR_BGR2HSV);
        cvtColor(roi_img, gray, COLOR_BGR2GRAY);
        vector<Mat> hsv_ch;
        split(hsv, hsv_ch);
        threshold(hsv_ch[1], BinaryMask, 180, 255, THRESH_BINARY);
        gray.setTo(255, BinaryMask);
    } else {
        gray = roi_img.clone();
    }

    Mat binary;
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 51, 17);
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
            if (mu.m00 > 100 && mu.m00 > max_area) { 
                max_area = (float)mu.m00;
                best_center = Point3f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00), (float)mu.m00);
            }
        }
    }
    return best_center;
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
    unsigned char* out_data, int out_width, int out_height)
{
    auto start = std::chrono::high_resolution_clock::now();

    Mat img(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, in_data);
    Mat out_img(out_height, out_width, CV_8UC3, out_data);
    
    vector<Point2f> current_corners;
    bool tracking_success = false;

    //阶段1急速局部追踪
    if (use_tracking && prev_corners.size() == 4) {
        tracking_success = true;
        int roi_size = 200; 
        for (int i = 0; i < 4; i++) {
            Point2f pt = prev_corners[i];
            int rx = std::max(0, (int)pt.x - roi_size / 2);
            int ry = std::max(0, (int)pt.y - roi_size / 2);
            int rw = std::min(width - rx, roi_size);
            int rh = std::min(height - ry, roi_size);
            Rect roi_rect(rx, ry, rw, rh);
            Mat roi_img = img(roi_rect);
            Point3f local_center = FindSingleMarkerInROI(roi_img, channels);
            if (local_center.z > 0) {
                current_corners.push_back(Point2f(local_center.x + rx, local_center.y + ry));
            } else {
                tracking_success = false;
                current_corners.clear();
                break;
            }
        }
    }

    // 阶段2全局盲扫兜底
    if (!tracking_success) {
        float scale = 1100.0f / (float)max(width, height);
        Mat small_img;
        resize(img, small_img, Size(), scale, scale, INTER_AREA);
        
        vector<Point3f> small_centers = FindMarkerCenters(small_img, channels);
        vector<Point3f> centers;
        for (auto pt : small_centers)
            centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));

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

            int br_idx = -1;
            float min_ratio = 1e9;
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
            for (int i = 0; i < 4; i++) {
                if (norm(sorted_pts[i] - br_point) < 1.0f) { s_br_idx = i; break; }
            }

            current_corners.push_back(sorted_pts[(s_br_idx + 2) % 4]); // TL
            current_corners.push_back(sorted_pts[(s_br_idx + 3) % 4]); // TR
            current_corners.push_back(sorted_pts[s_br_idx]);           // BR
            current_corners.push_back(sorted_pts[(s_br_idx + 1) % 4]); // BL
        }
    }

    //阶段3时域滤波与死区锁死(别乱动参数啊，我亲自调的)
    if (current_corners.size() == 4) {
        if (use_tracking) {
            float alpha = 0.95f; 
            float deadzone_radius = 1.0f; 

            for (int i = 0; i < 4; i++) {
                float dist = (float)norm(current_corners[i] - prev_corners[i]);
                if (dist < deadzone_radius) {
                    current_corners[i] = prev_corners[i];
                } else {
                    current_corners[i] = current_corners[i] * alpha + prev_corners[i] * (1.0f - alpha);
                }
            }
        }

        prev_corners = current_corners;
        use_tracking = true;

        Point2f tl = current_corners[0], tr = current_corners[1], br = current_corners[2], bl = current_corners[3];
        float pad_x = out_width * 0.05225f, pad_y = out_height * 0.05225f, correct = out_width * 0.016f; 
        
        vector<Point2f> src = { tl, tr, br, bl };
        vector<Point2f> dst = { 
            Point2f(pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x - correct, out_height - 1 - pad_y - correct), 
            Point2f(pad_x, out_height - 1 - pad_y) 
        };
        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(img, out_img, M, Size(out_width, out_height), INTER_NEAREST);
        
    } else {
        use_tracking = false;
        return false;
    }

    // =========================================================================
    //                              [性能监控模块]
    //                                后期可删除
    // =========================================================================
    auto end = std::chrono::high_resolution_clock::now();
    double current_duration = std::chrono::duration<double, std::milli>(end - start).count();
    
    total_time_ms += current_duration;
    frame_counter++;
    double average_duration = total_time_ms / (double)frame_counter;

    std::cout << ">>> [Warp Engine] Mode: " << (tracking_success ? "Tracking" : "Global") 
              << " | Current: " << std::fixed << std::setprecision(1) << current_duration << " ms"
              << " | Average: " << std::fixed << std::setprecision(2) << average_duration << " ms" 
              << " (" << frame_counter << " frames)" << "\n";

    return true;
}
}