#include <iostream>
#include <iterator>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>

#define IMAGE_ "C:/opencv-master/opencv/samples/data/aero1.jpg"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


int drawKeypointCircle (Mat& image, KeyPoint& Kpoint)
{
    Point2f point = Kpoint.pt;
    circle( image, point , 4, Scalar(0, 0, 255), 1, 8, 0 );
    return 1;
}


int main(int argc, char* argv[])
{
    VideoCapture srcVideo(0);
    if (!srcVideo.isOpened()) {
        cout << "File invalid!" << endl;
        return -1;

    }
    Mat input_image, copy;
   // namedWindow( "SURF result", WINDOW_AUTOSIZE );
    while (srcVideo.read(input_image))
    {

       // input_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
         copy = input_image.clone();


        double hessian_threshold = 2000;
        Ptr<Feature2D> surf_detector_obj;
        vector<KeyPoint> keypoints1;


        double t = (double)getTickCount();
        surf_detector_obj = SURF::create(hessian_threshold);
        surf_detector_obj->detect( input_image, keypoints1);

        vector<KeyPoint>::iterator keypoints1_iterator;
        keypoints1_iterator = keypoints1.begin();

        while (keypoints1_iterator != keypoints1.end())
        {
            drawKeypointCircle(copy, *keypoints1_iterator++);
        }

        t = ((double)getTickCount() - t)/getTickFrequency();
        cout << "Times passed in seconds: " << t << endl;

        imshow( "SURF result", copy );
        //char c = cvWaitKey(33);
        //if (c == 27) { // если нажата ESC - выходим
        //        break;
        //}
    }
    //waitKey();
    return 0;
}

