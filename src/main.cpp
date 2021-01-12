#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "feature_location.hpp"

#include "utils.hpp"
#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Declare the output variables
    Mat canny_edges_gray, im_hough_lines, im_hough_segments;

    // Loads an image
    Mat im_gray_1 = imread("data/saliere/1.jpg", IMREAD_GRAYSCALE);
    Mat im_BGR_1 = imread("data/saliere/1.jpg", IMREAD_COLOR);

    Mat im_gray_2 = imread("data/saliere/2.jpg", IMREAD_GRAYSCALE);
    Mat im_BGR_2 = imread("data/saliere/2.jpg", IMREAD_COLOR);


    // Vectors containing the points used for the calibration
    vector<vector<Point3f>> object_points_1;
    vector<vector<Point2f>> image_points_1;

    vector<vector<Point3f>> object_points_2;
    vector<vector<Point2f>> image_points_2;

    
    find_points_mire(im_gray_1, im_BGR_1, object_points_1, image_points_1, string("Image 1"));
    find_points_mire(im_gray_2, im_BGR_2, object_points_2, image_points_2, string("Image 2"));

    Mat M_int_1(3,4,CV_64F);
    Mat M_ext_1(4,4,CV_64F);
    Mat cameraMatrix_1(3,3,CV_64F);

    Mat M_int_2(3,4,CV_64F);
    Mat M_ext_2(4,4,CV_64F);
    Mat cameraMatrix_2(3,3,CV_64F);

    Calibrate(im_gray_1, im_BGR_1, object_points_1, image_points_1, cameraMatrix_1, M_int_1, M_ext_1, "Calibrage image 1");
    Calibrate(im_gray_2, im_BGR_2, object_points_2, image_points_2, cameraMatrix_2, M_int_2, M_ext_2, "Calibrage image 2");

    


    Mat imageo1, imageo2;
    extract_features(im_gray_1, im_gray_2, &imageo1, &imageo2, 1000);






    // //FAST(image, &keypointsD, threshold, true);
    // drawKeypoints(im_gray_1, feat, imageKey);

    // Ptr<StereoSGBM> BMState = cv::StereoSGBM::create(0, 8 * 16, 3, 200, 400, 0, 15, 7, 200, 2, StereoSGBM::MODE_HH);
    // BMState->compute(im_gray_1, im_gray_2, imageKey);
    //imageKey = (imageKey - min) * 255 / (max - min);
    //cv::normalize(imageKey, imageKey, 0, 256, CV_MMX);
    //normalize(imageKey, imageKey, 0, 255, NORM_MINMAX, CV_8U);
    imshow("keypoints", imageo1);

    imshow("keypoints2", imageo2);

    // // Flann needs the descriptors to be of type CV_32F
    // descriptors_1.convertTo(descriptors_1, CV_8UC1);
    // descriptors_2.convertTo(descriptors_2, CV_8UC1);

    // std::cout << descriptors_1.rows << ", " << descriptors_1.cols << std::endl;
    // std::cout << descriptors_2.rows << ", " << descriptors_2.cols << std::endl;
    // std::cout << "_________________________________" <<std::endl;

    // Mat im_out;
    // drawKeypoints(im_gray_1, key_points_1, im_out, Scalar(0,0,255));

    /*
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;


    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance < 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }
    
    vector< Point2f > obj;
    vector< Point2f > scene;


    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( key_points_1[ good_matches[i].queryIdx ].pt );
        scene.push_back( key_points_1[ good_matches[i].trainIdx ].pt );
    }


    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, RANSAC );
    // Use the Homography Matrix to warp the images
    cv::Mat result;
    warpPerspective(im_gray_1,result,H,Size(im_gray_1.cols+im_gray_2.cols,im_gray_1.rows));
    cv::Mat half(result,cv::Rect(0,0,im_gray_2.cols,im_gray_2.rows));
    im_gray_2.copyTo(half);
    imshow( "Result", result );
*/

    // // Feature matching
	// BFMatcher BF = BFMatcher(NORM_HAMMING);
	// std::vector<vector<DMatch> > matches;
	// BF.knnMatch(descriptors_1, descriptors_2, matches, 2);

	// std::vector<DMatch> match1;
	// std::vector<DMatch> match2;

	// for (int i = 0; i < matches.size(); i++)
	// {
	// 	match1.push_back(matches[i][0]);
	// 	match2.push_back(matches[i][1]);
	// }

    // Mat img_matches1, img_matches2;
	// drawMatches(im_gray_1, key_points_1, im_gray_2, key_points_2, match1, img_matches1);
	// drawMatches(im_gray_1, key_points_1, im_gray_2, key_points_2, match2, img_matches2);

	// imshow("test2", img_matches1);
	// imshow("test4", img_matches2);


    // imshow("out", im_out);
    // Wait and Exit
    waitKey(0);
    return 0;
}