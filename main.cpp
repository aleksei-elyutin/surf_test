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


int drawKeypointCircle (Mat& image, KeyPoint& Kpoint, Scalar color)
{
    Point2f point = Kpoint.pt;
    circle( image, point , 4, color , 1, 8, 0 );
    return 1;
}
int drawLineBetweenKeypoints (Mat& image, KeyPoint& Kpoint1, KeyPoint& Kpoint2, Scalar color)
{
    Point2f point1 = Kpoint1.pt;
    Point2f point2 = Kpoint2.pt;
    line( image,point1,point2, color, 1, LINE_8, 0);
    return 1;
}

int main(int argc, char* argv[])
{
    //* TODO: исправить говнокод
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
    //**


    Mat previous_frame, current_frame, gray_copy;
    Ptr<SURF> surf_detector_obj;
    namedWindow( "SURF result", WINDOW_AUTOSIZE);

    int max_hessian_threshold = 4000, current_hessian_threshold = 4000;
    createTrackbar( "Threshold", "SURF result", &current_hessian_threshold, max_hessian_threshold);

    surf_detector_obj = SURF::create(
                current_hessian_threshold,
                1, // nOctaves - число октав
                3, // nOctaveLayers - число уровней внутри каждой октавы
                false, // использовать расширенный дескриптор
                true); // не использовать вычисление ориентации);



    vector<KeyPoint>::iterator previous_frame_keypoints_iterator, current_frame_keypoints_iterator;
    FlannBasedMatcher matcher;

    srcVideo.read(previous_frame); //читаем первый кадр

    while (srcVideo.read(current_frame))
    {
        vector<KeyPoint> previous_frame_keypoints, current_frame_keypoints;
        UMat _current_frame_descriptors, _previous_frame_descriptors;
        Mat current_frame_descriptors = _current_frame_descriptors.getMat(ACCESS_RW),
            previous_frame_descriptors = _previous_frame_descriptors.getMat(ACCESS_RW);

       // previous_frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);
       //  copy = previous_frame.clone();
       //  cvtColor(previous_frame,gray_copy,COLOR_BGR2GRAY);


        double t = (double)getTickCount(); //Временная метка 1 ***

        surf_detector_obj->setHessianThreshold( current_hessian_threshold ); //установка значения порога гессиана

        surf_detector_obj->detectAndCompute(
                    previous_frame,
                    Mat(),
                    previous_frame_keypoints,
                    previous_frame_descriptors); //ОТ с дескрипторами для предыдущего кадра

        surf_detector_obj->detectAndCompute(
                    current_frame,
                    Mat(),
                    current_frame_keypoints,
                    current_frame_descriptors); // ОТ с дескрипторами для текущего кадра

        std::vector< DMatch > matches;
        matcher.match( previous_frame_descriptors, current_frame_descriptors, matches );

        std::vector<KeyPoint> previous_frame_matched_features;
        std::vector<KeyPoint> current_frame_matched_features;
        std::vector<Point2f> previous_frame_matched_points;
        std::vector<Point2f> current_frame_matched_points;

        for( size_t i = 0; i < matches.size(); i++ )
        {
           //-- Get the keypoints from the good matches
           previous_frame_matched_features.push_back( previous_frame_keypoints[ matches[i].queryIdx ] );
           previous_frame_matched_points.push_back( previous_frame_keypoints[ matches[i].queryIdx ].pt );
           //cout <<  matches[i].queryIdx << "     ";
           current_frame_matched_features.push_back( current_frame_keypoints[ matches[i].trainIdx ] );
           current_frame_matched_points.push_back( current_frame_keypoints[ matches[i].trainIdx ].pt );
           //cout <<  matches[i].trainIdx << "     " << endl;
        }
        cout << "Number of matches: "<<matches.size() << endl;
        Mat copy = current_frame.clone();

        Mat H = findHomography( current_frame_matched_points, previous_frame_matched_points, RANSAC );
        //perspectiveTransform(current_frame_matched_features, current_frame_matched_features, H);
        warpPerspective(current_frame, copy,H,Size(current_frame.cols, current_frame.rows));


         /* previous_frame_keypoints_iterator =  previous_frame_matched_features.begin();
          cout << "previous_frame_matched_features = " << previous_frame_matched_features.size() << endl;
          while (previous_frame_keypoints_iterator !=  previous_frame_matched_features.end())
          {
              drawKeypointCircle(copy, *previous_frame_keypoints_iterator++,  Scalar(128, 0 , 10));
          }
          current_frame_keypoints_iterator =  current_frame_matched_features.begin();
           cout << "current_frame_matched_features = " << current_frame_matched_features.size() << endl;
          while (current_frame_keypoints_iterator != current_frame_matched_features.end())
          {
              drawKeypointCircle(copy, *current_frame_keypoints_iterator++,  Scalar(0, 200, 128));
          }
            */
           /* for (size_t i = 0; i<previous_frame_matched_features.size(); i++ )
            {
                drawKeypointCircle(copy, previous_frame_matched_features[i],  Scalar(255, 0 , 10));
                drawKeypointCircle(copy, current_frame_matched_features[i],  Scalar(0, 255 , 10));
                drawLineBetweenKeypoints(copy, previous_frame_matched_features[i],current_frame_matched_features[i], Scalar (128, 128, 128));
            }*/


  /*
        previous_frame_keypoints_iterator = previous_frame_keypoints.begin();
        while (previous_frame_keypoints_iterator != previous_frame_keypoints.end())
        {
            drawKeypointCircle(copy, *previous_frame_keypoints_iterator++,  Scalar(128, 0 , 10));
        }
        current_frame_keypoints_iterator = current_frame_keypoints.begin();
        while (current_frame_keypoints_iterator != current_frame_keypoints.end())
        {
            drawKeypointCircle(copy, *current_frame_keypoints_iterator++,  Scalar(0, 200, 128));
        }*/

        t = ((double)getTickCount() - t)/getTickFrequency(); //Временная метка 2 ***
        cout << "Times passed in seconds: " << t << endl;

        imshow( "SURF result", copy );
        if ( cvWaitKey(33)  == 27 )  break; //ESC for exit

        previous_frame = copy.clone();


        }

    return 0;
}

