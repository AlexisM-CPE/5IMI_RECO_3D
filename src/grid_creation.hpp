#ifndef GRID_CREATION_HPP
#define GRID_CREATION_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

#define NO_COLOR 0
#define MAGENTA  1
#define YELLOW   2
#define CYAN     3
#define WHITE    4

void find_pos(cv::Mat HSV, std::vector<cv::Point2f> points);
int find_color(cv::Mat HSV, cv::Point2f p);

#endif