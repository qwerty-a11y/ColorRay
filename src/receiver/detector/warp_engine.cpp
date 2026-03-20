#include <iostream>
#include <opencv2/opencv.hpp>//环境自己配opencv_world42x.dll要同文件夹并且不能带d
#include <vector>
#include <cmath>
#include <algorithm>
//#include <immintrin.h>//我本来想使用simd加速没有数据依赖的循环，好像用不上？
//还挺像做线程分配的说是，好像也不行
using namespace cv;
using namespace std;
/*==============================================================================
                                    [识别模块]
        input:输入图像             ch:通道数             功能:寻找定位块        
*///============================================================================
vector<Point3f> FindMarkerCenters(const Mat& input, int ch) {                               //python传进来的格式是BGR
    Mat gray;//声明灰度图矩阵
    if (ch == 3) {                                                                          //是否三通道彩色图
        Mat hsv, BinaryMask;                                                                //声明"HSV色彩空间"矩阵与"黑白掩膜"矩阵
        cvtColor(input, hsv, COLOR_BGR2HSV);                                                //把BGR转成HSV色彩空间
        cvtColor(input, gray, COLOR_BGR2GRAY);                                              //把BGR转成单通道灰度图
        vector<Mat> hsv_ch;
        split(hsv, hsv_ch);                                                                 //拆分成"色相[0]","饱和度[1]","亮度[2]"
        threshold(hsv_ch[1], BinaryMask, 180, 255, THRESH_BINARY);                          //二值化操作，把hsv_ch[1]里饱和度超过180的区域转成255(纯白),THRESH_BINARY是二值化参数,现在我们把所有白色膜区域，全部在灰度图里全部变成白色，，其他的一律定成黑色，这样子整张图就只剩四角定位块和中间零散黑点        
        gray.setTo(255, BinaryMask);                                                        //把二值图的掩膜打在灰度图上,
    } else {  
        gray = input.clone();                                                               //黑白图直接用了
    }

    /*//========================================================================
    不要问我为啥定位块我不同意做彩色，自己看24行，我使用饱和度来判断，
    以上方案改不了一点，所有的参数和色彩空间已经无法进一步调优，我手搓
    好几种新方案均不如这种，没必要调优这一部分了，已经是极限了
    *///========================================================================

    Mat binary;                                                                             //声明局部二值化矩阵
    adaptiveThreshold(gray,binary,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_BINARY_INV,101,15); //高斯加权处理近似，不要改第二个参数
    Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2));                            //定义一个L形手术刀，在后面做单像素摩尔纹处理
    Mat closed_binary;                                                                      //声明闭运算处理后的黑白图像
    morphologyEx(binary, closed_binary, MORPH_OPEN, kernel);                               //开始做闭运算处理

    /*//========================================================================
    上面的模块就别删了，理论上是矩形开运算，因为我搞反了，因为我需要吃掉白色摩尔纹
    所以应该开运算，但是误打误撞，闭运算L型手术刀效果很好，各位活爹，上面改的代码和参数
    调了两节课，不敢动，能跑就行，各位活爹也都别动了
    *///========================================================================

    // =========================================================================
    //[调试窗口1]展示形态学缝合后的黑白反转二值图
    // static bool window_init = false;
    // if (!window_init) {
    //     namedWindow("Debug: Binarized", WINDOW_NORMAL);
    //     window_init = true;
    // }
    // imshow("Debug: Binarized", closed_binary);
    // =========================================================================

    vector<vector<Point>> contours;                                                         //储存轮廊点集合
    vector<Vec4i> hierarchy;                                                                //储存轮廊父子关系
    findContours(closed_binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);       //opencv函数，输入图像，轮廊点容器，轮廊关系容器，拓扑结构，轮廓近似方法
    
    /*//========================================================================
        mod
        RETR_EXTERNAL	只找最外层。
        RETR_LIST	    找所有轮廓，但不建立父子关系。
        RETR_CCOMP	    建立两层级。顶层是外框，第二层是里面的孔。	
        RETR_TREE	    建立完整树形结构，实际那个定位块会找到多维嵌套
        compu参数也换不了一点，上面模块也不要动了
    *///========================================================================

    vector<Point3f> centers;                                                                //存符合条件的中心点和它的面积
    for (size_t i=0;i<contours.size();i++) {
        //计录i的第一个子轮廊，所以我才说叫你们把定位块画大一点，不要一个个都说“变彩色不就好了嘛~~~~~”
        int kid_idx=hierarchy[i][2];
        int cnt = 0;//层数
        while(kid_idx!=-1) {//类似跳表，一路进去数几层，其实2层就能跳出来了
            cnt++;
            kid_idx=hierarchy[kid_idx][2];
            if(cnt>=2)break;//保证性能
        }
        if (cnt >= 2) {                                                                     //两层了，杀掉无嵌套噪点
            Moments mu = moments(contours[i], false);                                       //提取特征矩
            if (mu.m00 >= 700)                                                               //m00是面积矩，用来杀掉小白点干扰的
                centers.push_back(Point3f(mu.m10 / mu.m00, mu.m01 / mu.m00, mu.m00));       //计算质心，同时存下面积
        }
    }
    vector<Point3f> merged;//存最后的定位点
    //合并去重。防止距离过进产生多个定位点干扰
    for (const auto& pt : centers) {
        bool is_new = true;
        for (auto& m : merged) {
            if (norm(Point2f(pt.x, pt.y) - Point2f(m.x, m.y)) < 150.0) {                    //杀掉距离过近的定位快
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
    Mat img(height, width, (channels == 3) ? CV_8UC3 : CV_8UC1, in_data);                   //把py传进来的指针变成opencv能处理的mat对象，这里0拷贝，速度会快一点
    Mat out_img(out_height, out_width, CV_8UC3, out_data);                                  //彩色图
    float scale = 800.0f / max(width, height);                                              //缩放比例，杀摩尔纹，以及提高计算速度，4k图要找到什么时候，我只能搜小处理,然后超不多一张图片的时间消耗差不多是50ms
    Mat small_img;                                                                          //小图像                                                                                  
    resize(img, small_img, Size(), scale, scale, INTER_AREA);                               //ai协助的INTER_AREA：区域插值法，这是缩小图像时最能抗锯齿，抗摩尔纹的插值方式。
    vector<Point3f> small_centers = FindMarkerCenters(small_img, channels);                 //然后我们用识别模块，开始找
    vector<Point3f> centers;
    for (auto pt : small_centers) //c++17的绑定
        centers.push_back(Point3f(pt.x / scale, pt.y / scale, pt.z));

    // =========================================================================
    // [调试窗口2]定位块标记，取消注释即可观看标记
    // Scalar drawColor(0, 255, 0);
    // for (size_t i = 0; i < centers.size(); i++) {
    //     Point2f draw_pt(centers[i].x, centers[i].y);
    //     circle(img, draw_pt, 12, drawColor, 2);
    //     string label = to_string(i);
    //     putText(img, label, Point(draw_pt.x + 10, draw_pt.y + 10), 
    //             FONT_HERSHEY_SIMPLEX, 1.0, drawColor, 2);
    // }
    // =========================================================================
    
    Point2f tl, tr, br, bl;                                                                 //四角变量
    if (centers.size() >= 4) {                                            
        Point2f approx_center(0, 0);                                                        //近似中心误差排除法
        for(auto p : centers) approx_center += Point2f(p.x, p.y);
        approx_center.x /= centers.size(); approx_center.y /= centers.size();
        if (centers.size() > 4) {
            //距离近似中心的远近降序排列
            sort(centers.begin(), centers.end(), [&approx_center](Point3f a, Point3f b) {
                return norm(Point2f(a.x, a.y) - approx_center) > norm(Point2f(b.x, b.y) - approx_center);
            });
            //隔离中间噪点！
            centers.resize(4); 
        }
        //重新计算绝对中心！
        Point2f exact_center(0, 0);
        for(auto p : centers) exact_center += Point2f(p.x, p.y);
        exact_center.x /= 4.0f; exact_center.y /= 4.0f;
        //2.0新增全新抗透视反转模块：相似定理原理
        //如果透视导致近处的块变大，那么它到中心的距离也会按等比例拉长
        //用面积/距离的平方，就能抵消透视畸变
        //右下角比值永远是全场最小的
        int br_idx = -1;
        float min_ratio = 1e9;
        for (int i = 0; i < 4; i++) {
            float dist = norm(Point2f(centers[i].x, centers[i].y) - exact_center);
            float area = centers[i].z;//m00面积
            float ratio = area / (dist * dist); 
            if (ratio < min_ratio) {
                min_ratio = ratio;
                br_idx = i;//比值最小的点是BR
            }
        }
        Point2f br_point = Point2f(centers[br_idx].x, centers[br_idx].y);
        //围成一个凸四边形
        vector<Point2f> sorted_pts;
        for(auto p : centers) sorted_pts.push_back(Point2f(p.x, p.y));
        sort(sorted_pts.begin(), sorted_pts.end(), [&exact_center](Point2f a, Point2f b) {
            return atan2(a.y - exact_center.y, a.x - exact_center.x) < atan2(b.y - exact_center.y, b.x - exact_center.x);
        });
        int sorted_br_idx = 0;
        for(int i = 0; i < 4; i++) {
            if(norm(sorted_pts[i] - br_point) < 1.0f) { 
                sorted_br_idx = i;
                break;
            }
        }
        //2.0致命错误修复，补上了前面删错的代码
        br = sorted_pts[sorted_br_idx];
        bl = sorted_pts[(sorted_br_idx + 1) % 4]; 
        tl = sorted_pts[(sorted_br_idx + 2) % 4];
        tr = sorted_pts[(sorted_br_idx + 3) % 4]; 
        float pad_x = out_width * 0.05225; 
        float pad_y = out_height * 0.05225;

    //==========================================================================
    //                    [方向定位块拉伸变形修正模块]
        float correct = out_width * 0.0140f;                                                //我人工调的参数不要动了  
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

    //==========================================================================
    //[调试窗口3]                   [性能计算模块]
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << ">>> [Warp Engine] 处理耗时: " << duration_ms << " ms (" << duration_us << " us)" << std::endl;

    //==========================================================================
    return true;
}
}