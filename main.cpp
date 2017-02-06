#include <iostream>
#include <iterator>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


int drawKeypointCircle (Mat& image, KeyPoint& Kpoint)
{
    Point2f point = Kpoint.pt;
    circle( image, point , 4, Scalar(0, 200, 128), 1, 8, 0 );
    return 1;
}

int main(int argc, char* argv[])
{
    VideoCapture srcVideo;
    if (argc == 1) {
        srcVideo = VideoCapture(0);
            if (!srcVideo.isOpened()) {
                cout << "File invalid!" << endl;
                return -1;
            }
    }
    else {
         srcVideo = VideoCapture(argv[1]);
            if (!srcVideo.isOpened()) {
                cout << "File invalid!" << endl;
                return -1;

            }
      }



    Mat input_image, gray_copy, copy ;
    Ptr<SURF> surf_detector_obj;
    namedWindow( "SURF result", WINDOW_AUTOSIZE );

    int max_hessian_threshold = 4000, current_hessian_threshold = 1000;
    createTrackbar( "Threshold", "SURF result", &current_hessian_threshold, max_hessian_threshold);

    surf_detector_obj = SURF::create(
                current_hessian_threshold,
                1, // nOctaves - число октав
                3, // nOctaveLayers - число уровней внутри каждой октавы
                false, // использовать расширенный дескриптор
                true); // не использовать вычисление ориетнации);
    


    while (srcVideo.read(input_image))
    {
       // input_image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
         copy = input_image.clone();
         cvtColor(input_image,gray_copy,COLOR_BGR2GRAY);



        vector<KeyPoint> keypoints1;
        UMat _descriptors1, _descriptors2;
        Mat descriptors1 = _descriptors1.getMat(ACCESS_RW),
            descriptors2 = _descriptors2.getMat(ACCESS_RW);





        double t = (double)getTickCount(); //Временная метка 1 ***

        surf_detector_obj->setHessianThreshold( current_hessian_threshold );

        surf_detector_obj->detectAndCompute(
                    input_image,
                    Mat(),
                    keypoints1,
                    descriptors1);


        vector<KeyPoint>::iterator keypoints1_iterator;
        keypoints1_iterator = keypoints1.begin();

        while (keypoints1_iterator != keypoints1.end())
        {
            drawKeypointCircle(copy, *keypoints1_iterator++);
        }

        t = ((double)getTickCount() - t)/getTickFrequency(); //Временная метка 2 ***
        cout << "Times passed in seconds: " << t << endl;

        imshow( "SURF result", copy );
        if ( cvWaitKey(33)  == 27 )  break; //ESC for exit

        }

    return 0;
}

