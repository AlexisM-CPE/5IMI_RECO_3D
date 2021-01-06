#ifndef POINTS_DETECTION_HPP
#define POINTS_DETECTION_HPP

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <cmath>


// intersection between two segments defined by (o1, p1) and (o2, p2). The intersection point is saved in r.
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);

// Merges close points 
std::vector<cv::Point2f> merge_close_points(const std::vector<cv::Point2f> &p, float r);



#endif