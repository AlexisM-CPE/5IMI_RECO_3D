#include "Point_Image.hpp"

Point_Image::Point_Image() : Point_Image::Point_Image(cv::Point2f(0.0f, 0.0f))
{
}

Point_Image::Point_Image(cv::Point2f coord_pix_param) : Point_Image::Point_Image(coord_pix_param, 0, 0.0f, 0.0f, 0.0f)
{
}

Point_Image::Point_Image(cv::Point2f coord_pix_param, int color_param, float H_param, float S_param, float V_param) : coord_pix_data(coord_pix_param), color_data(color_param), H_data(H_param), S_data(S_param), V_data(V_param)
{
}

cv::Point2f const &Point_Image::get_coord_pix() const
{
    return coord_pix_data;
}

int const &Point_Image::get_color_int() const
{
    return color_data;
}

std::vector<float> const &Point_Image::get_HSV_vector() const
{
    return std::vector<float>{H_data, S_data, V_data};
}

float const &Point_Image::H() const
{
    return H_data;
}

float const &Point_Image::S() const
{
    return S_data;
}

float const &Point_Image::V() const
{
    return V_data;
}

void Point_Image::find_color_Point_Image(cv::Mat HSV)
{
    //cv::Vec3f hsv_color = HSV.at<cv::Vec3f>(coord_pix_data);

    int i = 0;
    float H_tot = 0.0f;
    float S_tot = 0.0f;
    float V_tot = 0.0f;

    for (float j = -10; j < 10; j++)
    {
        for (float k = -10; k < 10; k++)
        {
            cv::Point2f p_temp = coord_pix_data - cv::Point2f(j, k);
            if (is_in_img(p_temp))
            {
                if (HSV.at<cv::Vec3f>(p_temp)[2] > 0.44f)
                {
                    H_tot += HSV.at<cv::Vec3f>(p_temp)[0];
                    S_tot += HSV.at<cv::Vec3f>(p_temp)[1];
                    V_tot += HSV.at<cv::Vec3f>(p_temp)[2];
                    i++;
                }
            }
        }
    }

    H_data = H_tot / i;
    S_data = S_tot / i;
    V_data = V_tot / i;

    color_data = find_color(HSV, coord_pix_data);
}

std::vector<float> mean_color(std::vector<Point_Image> points_in_lines)
{
    float H_tot = 0.0f;
    float S_tot = 0.0f;
    float V_tot = 0.0f;
    uint i = 0;

    for (auto p : points_in_lines)
    {
        H_tot += p.H();
        S_tot += p.S();
        V_tot += p.V();
        i++;
    }
    return std::vector<float>{H_tot / i, S_tot / i, V_tot / i};
}