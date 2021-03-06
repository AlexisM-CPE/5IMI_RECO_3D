#ifndef POINTS_DETECTION_HPP
#define POINTS_DETECTION_HPP

#include "opencv2/highgui.hpp"

#include <iostream>
#include <cmath>


// intersection between two segments defined by (o1, p1) and (o2, p2). The intersection point is saved in r.
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);

// Merges close points 
std::vector<cv::Point2f> merge_close_points(const std::vector<cv::Point2f> &p, float r);

std::vector<int> kmeans(std::vector<float> const& angles, int it_max = 10, int k=2, float min_v = 0.0f, float max_v = CV_PI);
std::vector<float> get_angles(std::vector<cv::Vec4i> linesP);

#endif