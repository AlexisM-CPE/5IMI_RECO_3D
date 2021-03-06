#include "grid_creation.hpp"
#include <cmath>

using namespace cv;
using namespace std;

struct neighbour
{
    vector<pair<int, int>> case_color;
};

int find_color(Mat HSV, Point2f p)
{
    // --- Creation of the Magenta ---
    Mat magenta(1, 1, CV_32FC3);
    Mat HSV_magenta;
    magenta = Scalar(1.0f, 0, 1.0f);
    cvtColor(magenta, HSV_magenta, COLOR_BGR2HSV);
    // -------------------------------

    // --- Creation of the Yellow ---
    Mat yellow(1, 1, CV_32FC3);
    Mat HSV_yellow;
    yellow = Scalar(0.0f, 1.0f, 1.0f);
    cvtColor(yellow, HSV_yellow, COLOR_BGR2HSV);
    // -------------------------------

    // --- Creation of the Cyan ---
    Mat cyan(1, 1, CV_32FC3);
    Mat HSV_cyan;
    cyan = Scalar(1.0f, 1.0f, 0.0f);
    cvtColor(cyan, HSV_cyan, COLOR_BGR2HSV);
    // -------------------------------

    int i = 0;
    float H_tot = 0.0f;
    float S_tot = 0.0f;
    float V_tot = 0.0f;
    float H = 0;
    float S = 0;
    float V = 0;
    for (float j = -10; j < 10; j++)
    {
        for (float k = -10; k < 10; k++)
        {
            Point2f p_temp = p - Point2f(j, k);
            if (is_in_img(p_temp))
            {
                if (HSV.at<Vec3f>(p_temp)[2] > 0.44f)
                {
                    H_tot += HSV.at<Vec3f>(p_temp)[0];
                    S_tot += HSV.at<Vec3f>(p_temp)[1];
                    V_tot += HSV.at<Vec3f>(p_temp)[2];
                    i++;
                }
            }
        }
    }
    H = H_tot / i;
    S = S_tot / i;
    V = V_tot / i;
    if (i < 150)
    {
        return NO_COLOR;
    }
    else if (S < 0.06f)
    {
        if (V > 0.75f)
            return WHITE;
        else
            return NO_COLOR;
    }
    else if (abs(HSV_magenta.at<Vec3f>(0, 0)[0] - H) < 40.0f)
    {
        if ((V > 0.7f) && (S > 0.5f))
            return MAGENTA;
        else
            return NO_COLOR;
    }
    else if (abs(HSV_yellow.at<Vec3f>(0, 0)[0] - H) < 30.0f)
    {
        if ((V > 0.7f) && (S > 0.5f))
            return YELLOW;
        else
            return NO_COLOR;
    }
    else if (abs(HSV_cyan.at<Vec3f>(0, 0)[0] - H) < 35.0f)
    {
        if ((V > 0.7f) && (S > 0.7f))
            return CYAN;
        else
            return NO_COLOR;
    }
    else
    {
        return NO_COLOR;
    }
}

bool is_in_img(cv::Point2f p)
{
    return ((p.x >= 0) && (p.y >= 0) && (p.y < 846) && (p.x < 1504));
}

int find_dir(Point2f dir, case_dir &case_d, int loop)
{
    int dir_name = 0;
    float n = sqrt(pow(dir.x, 2) + pow(dir.y, 2));
    float cos = dir.y / n;
    float sin = dir.x / n;
    float angle = acosf(cos);
    if (-dir.x < 0)
        angle = -angle;
    angle = angle * 180 / M_PI;
    // std::cout << " cos : " << cos << " sin : " << sin << " norm : " << n << std::endl;
    // std::cout << " angle : " << angle << std::endl;
    if ((angle > 135) || (angle <= -135))
    {
        case_d.west = loop + 1;
        dir_name = WEST;
    }

    else if (angle >= 45)
    {
        case_d.north = loop + 1;
        dir_name = NORTH;
    }
    else if (angle >= -45)
    {
        case_d.east = loop + 1;
        dir_name = EAST;
    }
    else
    {
        case_d.south = loop + 1;
        dir_name = SOUTH;
    }

    return dir_name;
}

std::vector<Point_Mire *> find_pos(Mat HSV, vector<Point2f> points)
{
    std::vector<Point_Mire *> vector_mire;
    for (uint i = 0; i < points.size(); i++)
    {
        // For all points detected by Hough
        Point_Image p_image = Point_Image(points[i]);
        p_image.find_color_Point_Image(HSV); // Get HSV of the current point
        int c = p_image.get_color_int();     // Get color of the current point (Magenta, Yellow, Cyan, White or No Color);

        if (c != NO_COLOR)
        {
            // --- Get the fourth closest points ---
            vector<pair<float, int>> close_norm;
            unsigned int j = 0;
            for (auto p : points)
            {
                close_norm.push_back(make_pair(norm(p - p_image.get_coord_pix()), j));
                j++;
            }
            sort(close_norm.begin(), close_norm.end());
            close_norm.erase(close_norm.begin());
            close_norm.erase(close_norm.begin() + 4, close_norm.end());
            // --------------------------------------

            bool verif = true;
            // If The distance between all the points is correct
            if ((abs(close_norm[0].first - close_norm[1].first) > 5.0f) && (abs(close_norm[2].first - close_norm[3].first) > 5.0f))
                verif = false;

            // If all points have the same color
            if (!((find_color(HSV, points[close_norm[0].second]) == c) && (find_color(HSV, points[close_norm[1].second]) == c) && (find_color(HSV, points[close_norm[2].second]) == c) && (find_color(HSV, points[close_norm[3].second]) == c)))
                verif = false;

            if (verif)
            {
                // Init neighbour and case direction
                neighbour nghbr;
                if (!nghbr.case_color.empty())
                    nghbr.case_color[0] = make_pair(0, 0);
                case_dir case_d;

                for (int k = 0; k < 4; k++)
                {
                    Point2f dir = points[close_norm[k].second] - p_image.get_coord_pix();
                    int loop = 0;
                    bool stop = false;
                    int dir_name = 0;
                    int test_fa = 0;
                    int test_tr = 0;
                    int color_tr = 0;
                    bool false_find = false;
                    Point2f pc = p_image.get_coord_pix() + dir;
                    Point2f p_prev = p_image.get_coord_pix();
                    vector<pair<float, int>> close_point;
                    vector<Point_Image> points_in_lines;
                    points_in_lines.push_back(p_image);
                    vector<float> hsv_color;
                    vector<float> mean_c;
                    while (stop == false)
                    {
                        mean_c = mean_color(points_in_lines);                          // Get mean color of the current line
                        points_in_lines.push_back(Point_Image(pc));                    // Add the current point in the line
                        int idx_last_element = points_in_lines.size() - 1;             // Update last element index
                        points_in_lines[idx_last_element].find_color_Point_Image(HSV); // Calculate color and update HSV for the current point
                        // Get HSV vector
                        hsv_color = vector<float>{points_in_lines[idx_last_element].H(), points_in_lines[idx_last_element].S(), points_in_lines[idx_last_element].V()};
                        int color = points_in_lines[idx_last_element].get_color_int(); // Get color

                        // --- Verification if the current point color's is near the color of all the previous points in the line ---
                        if (p_image.get_color_int() == WHITE)
                        {
                            if ((abs(hsv_color[1] - mean_c[1]) > 0.02f) && (false_find == false))
                            {
                                false_find = true;
                                dir_name = find_dir(dir, case_d, loop);
                            }
                        }
                        else if (p_image.get_color_int() != WHITE)
                        {
                            if (((abs(hsv_color[0] - mean_c[0]) > 5.0f) || (abs(hsv_color[1] - mean_c[1]) > 0.1f)) && (false_find == false))
                            {
                                false_find = true;
                                dir_name = find_dir(dir, case_d, loop);
                            }
                        }
                        // -----------------------------------------------------------------------------------------------------------

                        if (color == NO_COLOR)
                        {
                            test_fa += 1; // Find false color + 1
                        }
                        else if (color == color_tr)
                        {
                            test_fa = 0; // Reset find false color
                        }
                        else if (color != c)
                        {
                            test_tr = 0;      // Reset find true color
                            color_tr = color; // Update the current color
                            test_fa += 1;     // Find false color + 1
                        }

                        if (test_tr >= 2)
                        {
                            stop = true;
                        }
                        if (test_fa >= 15)
                        {
                            stop = true;
                        }

                        // ---Change point---
                        p_prev = pc;
                        pc += dir;
                        // ------------------

                        if (!(is_in_img(pc)))
                        {
                            stop = true; // If the point isn't in the image
                        }
                        if (stop == false)
                        {
                            j = 0;
                            for (auto p : points)
                            {
                                close_point.push_back(make_pair(norm(p - pc), j));
                                j++;
                            }
                            sort(close_point.begin(), close_point.end());
                            // Verification if a hough point is close to th new current point
                            if (close_point[0].first < 7.0f)
                                pc = points[close_point[0].second];
                            close_point.erase(close_point.begin(), close_point.end());
                            dir = pc - p_prev;
                        }
                        loop += 1;
                    }
                    if (color_tr > NO_COLOR)
                    {
                        // Fill the neighbour struct with correct values
                        switch (dir_name)
                        {
                        case NORTH:
                            nghbr.case_color.push_back(make_pair(case_d.north, color_tr));
                            break;
                        case SOUTH:
                            nghbr.case_color.push_back(make_pair(case_d.south, color_tr));
                            break;
                        case EAST:
                            nghbr.case_color.push_back(make_pair(case_d.east, color_tr));
                            break;
                        case WEST:
                            nghbr.case_color.push_back(make_pair(case_d.west, color_tr));
                            break;
                        default:
                            break;
                        }
                    }
                }

                if ((case_d.north + case_d.south == 8) && (case_d.east + case_d.west == 8))
                {
                    // If the position (North, South, East, West) seems correct
                    if (nghbr.case_color.size() == 2)
                    {
                        // If there is only two colors around the point (Magenta, Yellow, Cyan, White)

                        int coord_x = 0; // Init x coord
                        int coord_y = 0; // Init y coord
                        switch (c)
                        {
                            // Fill coord_x and coord_y with correct values dependiong on the color
                        case MAGENTA:
                            if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                            {
                                coord_x = 8 - nghbr.case_color[0].first;
                                coord_y = 8 - nghbr.case_color[1].first;
                            }
                            else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                            {
                                coord_x = 8 - nghbr.case_color[1].first;
                                coord_y = 8 - nghbr.case_color[0].first;
                            }
                            break;
                        case YELLOW:
                            if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                            {
                                coord_x = 8 + nghbr.case_color[1].first;
                                coord_y = 8 + nghbr.case_color[0].first;
                            }
                            else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                            {
                                coord_x = 8 + nghbr.case_color[0].first;
                                coord_y = 8 + nghbr.case_color[1].first;
                            }
                            break;
                        case CYAN:
                            if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                            {
                                coord_x = 8 - nghbr.case_color[1].first;
                                coord_y = 8 + nghbr.case_color[0].first;
                            }
                            else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                            {
                                coord_x = 8 - nghbr.case_color[0].first;
                                coord_y = 8 + nghbr.case_color[1].first;
                            }
                            break;
                        case WHITE:
                            if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                            {
                                coord_x = 8 + nghbr.case_color[0].first;
                                coord_y = 8 - nghbr.case_color[1].first;
                            }
                            else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                            {
                                coord_x = 8 + nghbr.case_color[1].first;
                                coord_y = 8 - nghbr.case_color[0].first;
                            }
                            break;
                        default:
                            break;
                        }
                        //std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                        Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int()); // Creation of the point in the mire
                        p_mire->compute_coords();
                        vector_mire.push_back(p_mire); // Add the point in the vector_mire
                    }
                }
            }
        }
    }
    return vector_mire; // Return the vector
}