#include <iostream>
#include <iterator>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>

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



    Mat previous_frame, current_frame, gray_copy;
    Ptr<SURF> surf_detector_obj;
    namedWindow( "SURF result", WINDOW_AUTOSIZE );

    int max_hessian_threshold = 4000, current_hessian_threshold = 1000;
    createTrackbar( "Threshold", "SURF result", &current_hessian_threshold, max_hessian_threshold);

    surf_detector_obj = SURF::create(
                current_hessian_threshold,
                1, // nOctaves - число октав
                3, // nOctaveLayers - число уровней внутри каждой октавы
                false, // использовать расширенный дескриптор
                true); // не использовать вычисление ориентации);

    vector<KeyPoint> previous_frame_keypoints, current_frame_keypoints;
    UMat _current_frame_descriptors, _previous_frame_descriptors;
    Mat current_frame_descriptors = _current_frame_descriptors.getMat(ACCESS_RW),
        previous_frame_descriptors = _previous_frame_descriptors.getMat(ACCESS_RW);

    vector<KeyPoint>::iterator previous_frame_keypoints_iterator, current_frame_keypoints_iterator;
    FlannBasedMatcher matcher;

    srcVideo.read(previous_frame);
    surf_detector_obj->detectAndCompute(
                previous_frame,
                Mat(),
                previous_frame_keypoints,
                previous_frame_descriptors);

    while (srcVideo.read(current_frame))
    {
       // previous_frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
       //  copy = previous_frame.clone();
         cvtColor(previous_frame,gray_copy,COLOR_BGR2GRAY);


        double t = (double)getTickCount(); //Временная метка 1 ***

        surf_detector_obj->setHessianThreshold( current_hessian_threshold );

        surf_detector_obj->detectAndCompute(
                    previous_frame,
                    Mat(),
                    current_frame_keypoints,
                    current_frame_descriptors);

        std::vector< DMatch > matches;
        matcher.match( previous_frame_descriptors, current_frame_descriptors, matches );

        std::vector<Point2f> previous_frame_matched_features;
        std::vector<Point2f> current_frame_matched_features;

        for( size_t i = 0; i < matches.size(); i++ )
        {
           //-- Get the keypoints from the good matches
           previous_frame_matched_features.push_back( current_frame_keypoints[ matches[i].queryIdx ].pt );
           current_frame_matched_features.push_back( previous_frame_keypoints[ matches[i].trainIdx ].pt );
        }
        Mat mask, copy;
        Mat H = findHomography( current_frame_matched_features, previous_frame_matched_features, mask ,RANSAC );
        perspectiveTransform(current_frame_matched_features, current_frame_matched_features, H);
        warpPerspective(current_frame, copy,H,Size(current_frame.cols, current_frame.rows));



        /*previous_frame_keypoints_iterator = previous_frame_keypoints.begin();
        while (previous_frame_keypoints_iterator != previous_frame_keypoints.end())
        {
            drawKeypointCircle(copy, *previous_frame_keypoints_iterator++);
        }*/

        t = ((double)getTickCount() - t)/getTickFrequency(); //Временная метка 2 ***
        cout << "Times passed in seconds: " << t << endl;

        imshow( "SURF result", copy );
        if ( cvWaitKey(33)  == 27 )  break; //ESC for exit

        previous_frame = copy.clone();
        previous_frame_keypoints = current_frame_keypoints;
        previous_frame_descriptors = current_frame_descriptors;

        }

    return 0;
}

