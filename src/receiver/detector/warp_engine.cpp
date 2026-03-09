#include <iostream>
#include <opencv2/opencv.hpp>//环境自己配opencv_world42x.dll要同文件夹并且不能带d
#include <vector>
#include <cmath>
#include <algorithm>
//#include <immintrin.h>//我本来想使用simd加速没有数据依赖的循环，好像用不上？
//还挺像做线程分配的说是，好像也不行
using namespace cv;
using namespace std;

// 识别模块,寻找标记中心
// 【修改】：返回 Point3f，x和y是坐标，z用来存储这个定位块的面积(m00)作为置信度
vector<Point3f> FindMarkerCenters(const Mat& input_img, int channels) {
    Mat gray;
    //这一步判断输入图像是不是三通道彩色色图
    if (channels == 3) {
        //1把BGR色彩空间转换为HSV色彩空间，分离颜色和亮度。
        Mat hsv, saturationMask;
        //2将HSV分离为Hue(色相)Saturation(饱和度)Value(明度)三个单通道图像
        //因为我只要找定位块，你什么几种颜色和我没关系，我就用色相，不要说什么我色相不行，我这里只判黑白，8色识别是你们的工作
        cvtColor(input_img, hsv, COLOR_BGR2HSV);
        vector<Mat> hsv_channels;
        //对饱和度通道(hsv_channels[1])进行二值化
        split(hsv, hsv_channels);
        //这里是把所有饱和度大于180的颜色全部生成白色膜，以后不要问我为啥定位块我不同意做彩色，自己看这里!!!!
        threshold(hsv_channels[1], saturationMask, 180, 255, THRESH_BINARY);
        //转换成灰度图
        cvtColor(input_img, gray, COLOR_BGR2GRAY);
        //现在我们把所有白色膜区域，全部在灰度图里全部变成白色，，其他的一律定成黑色，这样子整张图就只剩四角定位块和中间零散黑点
        gray.setTo(255, saturationMask); 
    } else {
        //黑白图直接用了
        gray = input_img.clone();
    }
    //做局部二值化，因为光照不均，别拿一个手电筒反驳我，拿手电筒举例子说“啊~~~，抗反光没用~~~”，你觉得没用你来改
    Mat binary;
    //因为总是一边亮一边暗，全局的话会暗处全变黑，所以局部
    //THRESH_BINARY_INV将图像反相，黑变白，然后找白岛
    //41是局部阈值像素块大小，这个可以改
    //10是微调常数，也可以改
    adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 41, 10);
    //现在定义一个核心，大小为2*2，别乱改，我调了一下午从30找到2
    //| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
    //V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V 
    //                 不要乱动！！！！！！！！
    Mat kernel = getStructuringElement(MORPH_CROSS, Size(2, 2));
    //
    Mat closed_binary;
    //这步ai的主意，但是效果很好，就别删了
    /*
    大致原理就是形态学闭运算，高分辨率我们二值化后，定位块可能撕裂，因为他是白的，摩尔纹会撕裂他
    我们把它给缝起来，抗摩尔纹干扰，这样子定位框就是闭合的了，顺手杀噪点
    */

    //这一步是机密，别的组都想不到的，不要外传上下模块都不要外传，我们的锐度目前是最高的  
    morphologyEx(binary, closed_binary, MORPH_CLOSE, kernel);

    // 存储轮廓点的集合 (contours) 和轮廓层级结构 (hierarchy)，识别是ai想的
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    // 在二值图上寻找所有的白岛，所以你原来的彩图边框必须是白色的！！！！！别问我为啥不能是黑色
    // RETR_TREE：记录轮廓之间的“父子嵌套关系”。
    vector<Point3f> centers;//存符合条件的中心点和它的面积
    for (size_t i=0;i<contours.size();i++) {
        //计录i的第一个子轮廊，所以我才说叫你们把定位块画大一点，不要一个个都说“变彩色不就好了嘛~~~~~”
        int kid_idx=hierarchy[i][2];
        int cnt = 0;//层数
        while(kid_idx!=-1) {//类似跳表，一路进去数几层，其实2层就能跳出来了
            cnt++;
            kid_idx=hierarchy[kid_idx][2];
            if(cnt>=2)break;//保证性能
        }
        if (cnt >= 2) {//两层了，杀掉无嵌套噪点
            Moments mu = moments(contours[i], false);
            if (mu.m00 > 500) //计算标记快x，y坐标，这个数值只能往大的改，这个判断不能删
                // 【修改】：把面积 mu.m00 塞进 z 坐标里带出去
                centers.push_back(Point3f(mu.m10 / mu.m00, mu.m01 / mu.m00, mu.m00));
        }
    }
    vector<Point3f> merged;//存最后的定位点
    //合并去重。防止距离过进产生多个定位点干扰
    for (const auto& pt : centers) {
        bool is_new = true;
        for (auto& m : merged) {
            // 注意：距离只算 x 和 y
            if (norm(Point2f(pt.x, pt.y) - Point2f(m.x, m.y)) < 100.0) { 
                is_new = false; 
                // 距离太近融合时，保留面积更大的那个作为真身
                if (pt.z > m.z) m = pt;
                break; 
            }
        }
        if (is_new) merged.push_back(pt);
    }
    return merged;
}

extern "C" {
//接口
__declspec(dllexport) bool ExtractQRCode(
    unsigned char* in_data, int width, int height, int channels,
    unsigned char* out_data, int out_width, int out_height) 
{
    //把py传进来的指针变成opencv能处理的mat对象，这里0拷贝，速度会快一点
    Mat img(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, in_data);
    Mat out_img(out_height, out_width, CV_8UC3, out_data);//彩色图
    //缩放比例，杀摩尔纹，以及提高计算速度，4k图要找到什么时候，我只能搜小处理
    float scale = 800.0f / max(width, height);
    Mat small_img;//小图像
    //ai协助的INTER_AREA：区域插值法，这是缩小图像时最能抗锯齿，抗摩尔纹的插值方式。
    resize(img, small_img, Size(), scale, scale, INTER_AREA);
    //然后我们用识别模块，开始找
    vector<Point3f> small_centers = FindMarkerCenters(small_img, channels);
    //这个就是定位块中心了
    //接下来是图像拉伸，拉回正常的位置
    vector<Point3f> centers;
    for (auto pt : small_centers) //c++17的绑定
        centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));

//| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
//V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V     
    //调试模块，画定位点的，想看定位点取消注释
    Scalar drawColor(0, 255, 0);
    for (size_t i = 0; i < centers.size(); i++) {
        Point2f draw_pt(centers[i].x, centers[i].y);
        circle(img, draw_pt, 12, drawColor, 2);
        string label = to_string(i);
        putText(img, label, Point(draw_pt.x + 10, draw_pt.y + 10), 
                FONT_HERSHEY_SIMPLEX, 1.0, drawColor, 2);
    }

    Point2f tl, tr, br, bl;//四角变量
    if (centers.size() >= 4) {
        //【保留你写的无敌近似中心隔离法】
        Point2f approx_center(0, 0);
        for(auto p : centers) approx_center += Point2f(p.x, p.y);
        approx_center.x /= centers.size(); approx_center.y /= centers.size();

        if (centers.size() > 4) {
            // 将点按照距离近似中心的远近降序排列（越远的排越前）
            sort(centers.begin(), centers.end(), [&approx_center](Point3f a, Point3f b) {
                return norm(Point2f(a.x, a.y) - approx_center) > norm(Point2f(b.x, b.y) - approx_center);
            });
            // 因为二维码定位块在四个角，距离中心最远，所以只保留前 4 个点，完美隔离中间噪点！
            centers.resize(4); 
        }

        // 既然现在只剩下纯正的 4 个定位块了，重新计算绝对精准的中心！
        Point2f exact_center(0, 0);
        for(auto p : centers) exact_center += Point2f(p.x, p.y);
        exact_center.x /= 4.0f; exact_center.y /= 4.0f;

        // =================================================================
        // 【全新抗透视反转模块】：透视尺度不变特征 (Scale-Invariant Relative Area)
        // 原理：如果透视导致近处的块变大，那么它到中心的距离也会按等比例拉长。
        // 用 "面积 / (距离的平方)"，就能抵消透视畸变！
        // 因为右下角(BR)物理面积最小，它的这个比值永远是全场最小的！
        int br_idx = -1;
        float min_ratio = 1e9;
        for (int i = 0; i < 4; i++) {
            float dist = norm(Point2f(centers[i].x, centers[i].y) - exact_center);
            float area = centers[i].z; // z里面存的就是m00面积
            float ratio = area / (dist * dist); 
            if (ratio < min_ratio) {
                min_ratio = ratio;
                br_idx = i; // 锁定比值最小的那个点，绝对是BR！
            }
        }
        Point2f br_point = Point2f(centers[br_idx].x, centers[br_idx].y);
        // =================================================================

        // 将四个点按顺时针排列围成一个凸四边形（环形排序）
        vector<Point2f> sorted_pts;
        for(auto p : centers) sorted_pts.push_back(Point2f(p.x, p.y));
        sort(sorted_pts.begin(), sorted_pts.end(), [&exact_center](Point2f a, Point2f b) {
            return atan2(a.y - exact_center.y, a.x - exact_center.x) < atan2(b.y - exact_center.y, b.x - exact_center.x);
        });

        // 既然排好序了，我们就看死锁的 br_point 在环形里的哪一个位置
        int sorted_br_idx = 0;
        for(int i = 0; i < 4; i++) {
            if(norm(sorted_pts[i] - br_point) < 1.0f) { // 加一层浮点数安全比较
                sorted_br_idx = i;
                break;
            }
        }

        // OpenCV中 atan2 顺时针转一圈的绝对顺序是：左上(TL) -> 右上(TR) -> 右下(BR) -> 左下(BL)
        // 既然知道哪一个是 BR 了，顺藤摸瓜，其余三个角瞬间落位！（不论图像怎么旋转都有效）
        
        // 【致命错误修复】：补上了这行原本漏掉的代码！！！
        br = sorted_pts[sorted_br_idx];
        
        bl = sorted_pts[(sorted_br_idx + 1) % 4]; // BR 的下一个是 BL
        tl = sorted_pts[(sorted_br_idx + 2) % 4]; // BR 的对角是 TL
        tr = sorted_pts[(sorted_br_idx + 3) % 4]; // BR 的上一个是 TR

        // 接下来透视变换会自动将图像旋转、翻转回正确的朝向。
        float pad_x = out_width * 0.05225; 
        float pad_y = out_height * 0.05225;
        //| | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
        //V V V V V V V V V V V V V V V V V V V V V V V V V V V V V V 
        // 你的神级参数保留！因为你的物理中心确实不是标准的正方形，会有微小的内推需求
        float correct = out_width * 0.0160f; //谁动这个参数我砍谁，我调了一下午，第四个定位块中心不准变
        vector<Point2f> src = { tl, tr, br, bl };
        vector<Point2f> dst = { 
            Point2f(pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x, pad_y), 
            Point2f(out_width - 1 - pad_x - correct, out_height - 1 - pad_y - correct), 
            Point2f(pad_x, out_height - 1 - pad_y) 
        };
        Mat M = getPerspectiveTransform(src, dst);
        warpPerspective(img, out_img, M, Size(out_width, out_height), INTER_NEAREST);
    } 
    return true;
}
}