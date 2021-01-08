#ifndef POINT_IMAGE_HPP
#define POINT_IMAGE_HPP

#include <opencv2/imgproc/imgproc.hpp>
#include "grid_creation.hpp"
#include <vector>

class Point_Image
{
public:
    Point_Image();
    Point_Image(cv::Point2f coord_pix_param);
    Point_Image(cv::Point2f coord_pix_param, int color_param, float H_param, float S_param, float V_param);

    // Getters
    cv::Point2f const &get_coord_pix() const;
    int const &get_color_int() const;
    std::vector<float> const &get_HSV_vector() const;
    float const &H() const;
    float const &S() const;
    float const &V() const;

    // Setters
    void set_color_int(int color_param);
    void H(float H);
    void S(float S);
    void V(float V);

    void find_color_Point_Image(cv::Mat HSV);

private:
    cv::Point2f coord_pix_data; // Coords in pix in the image
    int color_data;             // number of the color -> 1 : magenta, 2 : yellow, 3 : cyan, 4 : white, 0 : undefined
    float H_data;               // Average Hue around the pixel
    float S_data;               // Average Saturation around the pixel
    float V_data;               // Average Value around the pixel
};

std::vector<float> mean_color(std::vector<Point_Image> points_in_lines);
#endif