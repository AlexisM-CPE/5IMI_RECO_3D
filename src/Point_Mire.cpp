#include "Point_Mire.hpp"

#define grid_size 20.0f


Point_Mire::Point_Mire()
    :index_mire_data(Point2i(0,0), coord_pix_data(Point2f(0.0f, 0.0f), coord_obj_data(Point2f(0.0f, 0.0f)), color_data(0){}

Point_Mire::Point_Mire(cv::Point2i index_mire_param, cv::Point2f coord_pix_param)
    :index_mire_data(index_mire_param, coord_pix_data(coord_pix_param, coord_obj_data(Point2f(0.0f, 0.0f)), color_data(0)
    {
        compute_coords();
    }

Point_Mire::Point_Mire(cv::Point2i index_mire_param, cv::Point2f coord_pix_param, int color_param);
    :index_mire_data(index_mire_param, coord_pix_data(coord_pix_param, coord_obj_data(Point2f(0.0f, 0.0f)), color_data(color_param)
    {
        compute_coords();
    }


cv::Point2i const& Point_Mire::get_index_mire()
{
    return index_mire_data;
}

cv::Point2f const& Point_Mire::get_coord_pix()
{
    return coord_pix_data;
}

cv::Point2f const& Point_Mire::get_coord_obj()
{
    return coord_obj_data;
}

int const& Point_Mire::get_color_int()
{
    return color_data;
}

void Point_Mire::compute_coords()
{
    coord_obj_data = Point2f(index_mire_data.x * grid_size, index_mire_data.y * grid_size);
}



std::vector<std::vector<cv::Point3f>> extract_object_points(std::vector<cv::Point_Mire*> points_grille)
{
    std::vector<cv::Point3f> vec;
    for (Point_Mire* p : points_grille)
    {
        vec.push_back(p->get_coord_obj());
    }
    std::vector<std::vector<cv::Point3f>> output;
    output.push_back(vec);
    return output;
}




std::vector<std::vector<cv::Point2f>> extract_image_points(std::vector<cv::Point_Mire*> points_grille)
{
    std::vector<cv::Point2f> vec;
    for (Point_Mire* p : points_grille)
    {
        vec.push_back(p->get_coord_pix());
    }
    std::vector<std::vector<cv::Point2f>> output;
    output.push_back(vec);
    return output;
}
