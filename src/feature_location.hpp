#ifndef FEATURE_LOCATION_HPP
#define FEATURE_LOCATION_HPP

#include "opencv2/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include <iostream>
#include <cmath>


cv::Mat create_M_int(cv::Mat cameraMatrix);
cv::Mat create_M_ext(std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs);

cv::Mat compute_transition_matrix(cv::Mat M_int, cv::Mat M_ext);
cv::Point3f get_camera_position(std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs);

cv::Point3f image_to_grid_plan(cv::Point2f point_image, cv::Mat M_transition);
cv::Point3f find_feature_3d();

#endif