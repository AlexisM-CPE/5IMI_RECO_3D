#ifndef POINT_MIRE_HPP
#define POINT_MIRE_HPP

#include <opencv2/imgproc/imgproc.hpp>

#include <vector>


class Point_Mire
{
    public:
        Point_Mire();
        Point_Mire(cv::Point2i index_mire_param, cv::Point2f coord_pix_param);
        Point_Mire(cv::Point2i index_mire_param, cv::Point2f coord_pix_param, int color_param);

        cv::Point2i const& get_index_mire();
        cv::Point2f const& get_coord_pix();
        cv::Point2f const& get_coord_obj();
        int const& get_color_int();

        void compute_coords();


    private:
        cv::Point2i index_mire_data;    // index in the grid
        cv::Point2f coord_pix_data;     // Coords in pix in the image
        cv::Point2f coord_obj_data;     // Coords in mm in the object coordinate system
        int color_data;                 // number of the color -> 1 : magenta, 2 : yellow, 3 : cyan, 4 : white, 0 : undefined
    
}

std::vector<std::vector<cv::Point3f>> extract_object_points(std::vector<cv::Point_Mire> points_grille);
std::vector<std::vector<cv::Point2f>> extract_image_points(std::vector<cv::Point_Mire> points_grille);




#endif