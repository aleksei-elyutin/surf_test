#include <iostream>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"

#define IMAGE_ "C:/opencv-master/opencv/samples/data/aero1.jpg"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main()
{
    Mat input_image, copy;
    input_image = imread(IMAGE_, CV_LOAD_IMAGE_COLOR);
    copy = input_image.clone();


    double hessian_threshold = 2000;
    Ptr<Feature2D> surf_detector_obj;
    vector<KeyPoint> keypoints1;

    double t = (double)getTickCount();
    surf_detector_obj = SURF::create(hessian_threshold);
    surf_detector_obj->detect( input_image, keypoints1);
    for( size_t i = 0; i < keypoints1.size(); i++ )
    {
        KeyPoint current_kp;
        Point2f kp_coords;
        current_kp = keypoints1[i];
        kp_coords = current_kp.pt;
        circle( copy, kp_coords , 4, Scalar(0, 0, 255), 1, 8, 0 );
    }

    t = ((double)getTickCount() - t)/getTickFrequency();
    cout << "Times passed in seconds: " << t << endl;
    namedWindow( "SURF result", WINDOW_AUTOSIZE );
    imshow( "SURF result", copy );
    waitKey();
    return 0;
}

