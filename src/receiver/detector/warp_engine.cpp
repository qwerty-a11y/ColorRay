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
            if (src_ptr[x] * count <= s - C * count) {
                dst_ptr[x] = 255;
            } else {
                dst_ptr[x] = 0;
            }
        }
    }
}
vector<Point3f> FindMarkerCenters(const Mat& input, int ch) {
    Mat gray;
    if (ch == 3) {
        cvtColor(input, gray, COLOR_BGR2GRAY);
    } else {
        gray = input.clone();
    }
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
Point3f FindSingleMarkerInROI(const Mat& roi_img, int ch) {
    Mat gray;
    if (ch == 3) {
        cvtColor(roi_img, gray, COLOR_BGR2GRAY);
    } else {
        gray = roi_img.clone();
    }
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
            if (mu.m00 > 100 && mu.m00 > max_area) { 
                max_area = (float)mu.m00;
                best_center = Point3f((float)(mu.m10 / mu.m00), (float)(mu.m01 / mu.m00), (float)mu.m00);
            }
        }
    }
    return best_center;
}
extern "C" {
static vector<Point2f> prev_corners; 
static bool use_tracking = false;
static double total_time_ms = 0;
static long long frame_counter = 0;
static bool thread_pool_init = false;
__declspec(dllexport) bool ExtractQRCode(
    unsigned char* in_data, int width, int height, int channels,
    unsigned char* out_data, int out_width, int out_height)
{
    // 记录开始时间
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
            
            if (local_center.z > 0) {
                current_corners[i] = Point2f(local_center.x + rx, local_center.y + ry);
            } else {
                track_failed_flag = true;
            }
        }
        if (track_failed_flag) {
            tracking_success = false;
            current_corners.clear();
        }
    }
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
    auto end = std::chrono::high_resolution_clock::now();
    double current_duration = std::chrono::duration<double, std::milli>(end - start).count();
    total_time_ms += current_duration;
    frame_counter++;
    double average_duration = total_time_ms / (double)frame_counter;
    std::cout << ">>> [CPU Warp Engine] Mode: " << (tracking_success ? "Tracking" : "Global") 
              << " | Current: " << std::fixed << std::setprecision(1) << current_duration << " ms"
              << " | Average: " << std::fixed << std::setprecision(2) << average_duration << " ms" 
              << " (" << frame_counter << " frames)" << "\n";
    return true;
}
}