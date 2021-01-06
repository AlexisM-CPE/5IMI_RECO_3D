#ifndef GRID_CREATION_HPP
#define GRID_CREATION_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

#define NO_COLOR 0
#define MAGENTA  1
#define YELLOW   2
#define CYAN     3
#define WHITE    4

void find_pos(Mat HSV,vector<Point2f> points);
int find_color(Mat HSV, Point2f p);

#endif