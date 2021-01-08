#ifndef GRID_CREATION_HPP
#define GRID_CREATION_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include "Point_Image.hpp"
#include "Point_Mire.hpp"

#define NO_COLOR 0
#define MAGENTA 1
#define YELLOW 2
#define CYAN 3
#define WHITE 4

#define NORTH 1
#define SOUTH 2
#define EAST 3
#define WEST 4

struct case_dir
{
    int north;
    int south;
    int east;
    int west;
};

std::vector<Point_Mire *> find_pos(cv::Mat HSV, std::vector<cv::Point2f> points);
int find_color(cv::Mat HSV, cv::Point2f p);
int find_dir(cv::Point2f dir, struct case_dir &case_d, int loop);
bool is_in_img(cv::Point2f p);

#endif