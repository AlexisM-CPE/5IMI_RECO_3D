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

    waitKey(0);
    return 0;
}