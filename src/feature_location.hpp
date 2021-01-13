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
cv::Point3f get_camera_position(cv::Mat M_ext);

cv::Point3f image_to_grid_plan(cv::Point2f point_image, cv::Mat M_transition);
cv::Point3f find_intersection(cv::Point3f feature_world_2d_1, cv::Point3f cam_proj_1, cv::Point3f feature_world_2d_2, cv::Point3f cam_proj_2, float& t);
std::vector<cv::Point3f> find_feature_3d_im1_im2(std::vector<cv::Point2f> features_im1, std::vector<cv::Point2f> features_im2, cv::Point3f cam_pos_1, cv::Point3f cam_pos_2, cv::Mat M_transition_1, cv::Mat M_transition_2);

#endif