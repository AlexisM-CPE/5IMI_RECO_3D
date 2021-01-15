#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include "feature_location.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>



#include <iostream>
#include <cmath>


int find_points_mire(cv::Mat& im_gray, cv::Mat& im_BGR, std::vector<std::vector<cv::Point3f>>& object_points, std::vector<std::vector<cv::Point2f>>& image_points, std::string name="None");

void Calibrate(cv::Mat& im_gray, cv::Mat& im_BGR, std::vector<std::vector<cv::Point3f>>& object_points, std::vector<std::vector<cv::Point2f>>& image_points, cv::Mat& cameraMatrix, cv::Mat distCoeffs, cv::Mat& M_int, cv::Mat& M_ext, std::string name="None");

void create_cloud_file(std::vector<cv::Point3f> points, std::string filename);

std::vector<cv::Point3f> read_cloud_file(std::string filename);