#include <iostream>
#include <iterator>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/videostab.hpp>

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

int addImFromMask (Mat& input_image1, Mat& input_image2, Mat& output_image, Mat& mask)
{
   if ( (input_image1.rows != input_image2.rows) || (input_image1.rows != mask.rows) || (input_image2.rows != mask.rows) ||
        (input_image1.cols != input_image2.cols) || (input_image1.rows != mask.rows) || (input_image2.rows != mask.rows) )
   {
       cout << "addImFromMask: Arrays must be same lengths" << endl;
       return -1;
   }
   if (input_image1.type()!=input_image2.type())
   {
       cout << "addImFromMaks: Arrays must the same type" << endl;
       cout << input_image1.type() << " != " << input_image2.type() << endl;
       return -1;
   }
   output_image = input_image1.clone();

    int rows = mask.rows, cols =  mask.rows;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (mask.at<Vec3b>(i,j)[0]&mask.at<Vec3b>(i,j)[1]&mask.at<Vec3b>(i,j)[2]) {
                {
                        output_image.at<Vec3b>(i,j)[0]= input_image2.at<Vec3b>(i,j)[0];
                        output_image.at<Vec3b>(i,j)[1]= input_image2.at<Vec3b>(i,j)[1];;
                        output_image.at<Vec3b>(i,j)[2]= input_image2.at<Vec3b>(i,j)[2];;
                }
            }
        }
    }
    return 1;
}

Mat createAffineMatrix (Mat h)
{
    Mat tmp(2,3, h.type());
    tmp.at<double>(0,0) = h.at<double>(0,0);
    tmp.at<double>(0,2) = h.at<double>(0,1);
    tmp.at<double>(0,3) = h.at<double>(0,3);
    tmp.at<double>(1,1) = h.at<double>(1,1);
    tmp.at<double>(1,2) = h.at<double>(1,2);
    tmp.at<double>(1,3) = h.at<double>(1,3);
    return tmp;

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



    Ptr<SURF> surf_detector_obj;
    namedWindow( "SURF result", WINDOW_AUTOSIZE);
    namedWindow( "Source", WINDOW_AUTOSIZE);

    int max_hessian_threshold = 4000, current_hessian_threshold = 2000;
    createTrackbar( "Threshold", "SURF result", &current_hessian_threshold, max_hessian_threshold);

    surf_detector_obj = SURF::create(
                current_hessian_threshold,
                1, // nOctaves - число октав
                3, // nOctaveLayers - число уровней внутри каждой октавы
                false, // использовать расширенный дескриптор
                true); // не использовать вычисление ориентации);



    FlannBasedMatcher matcher;

    srcVideo.read(previous_frame); //читаем первый кадр


    //int cnt=0;

    while (srcVideo.read(current_frame))
    {
        double t = (double)getTickCount(); //Временная метка 1 ***

        vector<KeyPoint> previous_frame_keypoints, current_frame_keypoints;
        UMat _current_frame_descriptors, _previous_frame_descriptors;
        Mat current_frame_descriptors = _current_frame_descriptors.getMat(ACCESS_RW),
            previous_frame_descriptors = _previous_frame_descriptors.getMat(ACCESS_RW);

        mask = Mat::zeros(Size(current_frame.cols, current_frame.rows),current_frame.type());
        Mat warped_mask = mask.clone();

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

        //std::vector<KeyPoint> previous_frame_matched_features;
        //std::vector<KeyPoint> current_frame_matched_features;
        std::vector<Point2f> previous_frame_matched_points;
        std::vector<Point2f> current_frame_matched_points;

        for( size_t i = 0; i < matches.size(); i++ )
        {
           //previous_frame_matched_features.push_back( previous_frame_keypoints[ matches[i].queryIdx ] );
           previous_frame_matched_points.push_back( previous_frame_keypoints[ matches[i].queryIdx ].pt );
           //current_frame_matched_features.push_back( current_frame_keypoints[ matches[i].trainIdx ] );
           current_frame_matched_points.push_back( current_frame_keypoints[ matches[i].trainIdx ].pt );
        }
        cout << "Number of matches: "<<matches.size() << endl;
        Mat copy = current_frame.clone();

        if (current_frame_matched_points.size()&previous_frame_matched_points.size())
        {

             Mat gmotion = videostab::estimateGlobalMotionRansac(previous_frame_matched_points, current_frame_matched_points);
             cout << "etimated matrix: " << endl << gmotion << endl;
             Mat H = findHomography( current_frame_matched_points, previous_frame_matched_points, RANSAC );
             cout << "homography matrix: " << endl << H << endl;
             Mat H_affine = createAffineMatrix(gmotion);
             cout << "affine: " << endl << H_affine << endl;

             warpAffine(current_frame, copy, H_affine, Size(current_frame.cols, current_frame.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
             warpAffine(mask, warped_mask , H_affine , Size(mask.cols, mask.rows),INTER_LINEAR, BORDER_CONSTANT, Scalar(255,255,255));
             if (!addImFromMask(copy, previous_frame, copy, warped_mask))
             {
                 cout << "err..." << endl;
                 return -1;
             }


        }



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
        /*
        vector<KeyPoint>::iterator previous_frame_keypoints_iterator, current_frame_keypoints_iterator;
        for (size_t i = 0; i<previous_frame_matched_features.size(); i++ )
        {
            drawKeypointCircle(copy, previous_frame_matched_features[i],  Scalar(255, 0 , 10));
            drawKeypointCircle(copy, current_frame_matched_features[i],  Scalar(0, 255 , 10));
            drawLineBetweenKeypoints(copy, previous_frame_matched_features[i],current_frame_matched_features[i], Scalar (128, 128, 128));
        }

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



        //Mat preview; double alpha = 0.4;
        //addWeighted( copy, 0.5, previous_frame, 0.5, 0.0, copy);

        imshow( "SURF result", copy );
        imshow( "Source", current_frame );
        previous_frame = copy.clone();

        t = ((double)getTickCount() - t)/getTickFrequency(); //Временная метка 2 ***
        if ( cvWaitKey(33)  == 27 )  break;

        cout << "Framerate: " << 1/t << endl;



        }

    return 0;
}

