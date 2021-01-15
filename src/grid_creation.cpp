#include "grid_creation.hpp"

using namespace cv;
using namespace std;

struct neighbour
{
    vector<pair<int, int>> case_color;
};

int find_color(Mat HSV, Point2f p)
{
    Mat magenta(1, 1, CV_32FC3);
    Mat HSV_magenta;
    magenta = Scalar(1.0f, 0, 1.0f);
    cvtColor(magenta, HSV_magenta, COLOR_BGR2HSV);

    Mat yellow(1, 1, CV_32FC3);
    Mat HSV_yellow;
    yellow = Scalar(0.0f, 1.0f, 1.0f);
    cvtColor(yellow, HSV_yellow, COLOR_BGR2HSV);

    Mat cyan(1, 1, CV_32FC3);
    Mat HSV_cyan;
    cyan = Scalar(1.0f, 1.0f, 0.0f);
    cvtColor(cyan, HSV_cyan, COLOR_BGR2HSV);
    //Vec3f hsv_color = HSV.at<Vec3f>(p);

    int i = 0;
    float H_tot = 0.0f;
    float S_tot = 0.0f;
    float V_tot = 0.0f;
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
    float H = H_tot / i;
    float S = S_tot / i;
    float V = V_tot / i;
    if (i < 150)
    {
        return NO_COLOR;
    }
    else if (S < 0.10f)
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
    float cos = dir.y / norm(dir);
    float sin = dir.x / norm(dir);
    if (abs(cos) > abs(sin))
    {
        if (cos > 0)
        {
            case_d.east = loop + 1;
            dir_name = EAST;
        }
        else
        {
            case_d.west = loop + 1;
            dir_name = WEST;
        }
    }
    else
    {
        if (sin > 0)
        {
            case_d.north = loop + 1;
            dir_name = NORTH;
        }
        else
        {
            case_d.south = loop + 1;
            dir_name = SOUTH;
        }
    }
    return dir_name;
}

std::vector<Point_Mire *> find_pos(Mat HSV, vector<Point2f> points)
{
    int b = 0;
    std::vector<Point_Mire *> vector_mire;
    for (uint i = 0; i < points.size(); i++)
    {
        Point_Image p_image = Point_Image(points[i]);
        p_image.find_color_Point_Image(HSV);
        int c = p_image.get_color_int();
        if (c != NO_COLOR)
        {
            vector<pair<float, int>> close_norm;
            int j = 0;
            for (auto p : points)
            {
                close_norm.push_back(make_pair(norm(p - p_image.get_coord_pix()), j));
                j++;
            }
            sort(close_norm.begin(), close_norm.end());
            close_norm.erase(close_norm.begin());
            close_norm.erase(close_norm.begin() + 4, close_norm.end());
            bool verif = true;
            if (abs(close_norm[0].first - close_norm[3].first) > 25.0f)
                verif = false;

            if (!((find_color(HSV, points[close_norm[0].second]) == c) && (find_color(HSV, points[close_norm[1].second]) == c) && (find_color(HSV, points[close_norm[2].second]) == c) && (find_color(HSV, points[close_norm[3].second]) == c)))
                verif = false;
            if (verif)
            {
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

                    while (stop == false)
                    {
                        vector<float> mean_c = mean_color(points_in_lines);

                        points_in_lines.push_back(Point_Image(pc));
                        int idx_last_element = points_in_lines.size() - 1;
                        points_in_lines[idx_last_element].find_color_Point_Image(HSV);
                        vector<float> hsv_color = vector<float>{points_in_lines[idx_last_element].H(), points_in_lines[idx_last_element].S(), points_in_lines[idx_last_element].V()};
                        int color = points_in_lines[idx_last_element].get_color_int();
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
                            if (((abs(hsv_color[0] - mean_c[0]) > 2.0f) || (abs(hsv_color[1] - mean_c[1]) > 0.08f)) && (false_find == false))
                            {
                                false_find = true;
                                dir_name = find_dir(dir, case_d, loop);
                            }
                        }

                        if (color == NO_COLOR)
                        {
                            test_fa += 1;
                        }
                        else if (color == color_tr)
                        {
                            test_tr += 1;
                            test_fa = 0;
                        }
                        else if (color != c)
                        {
                            test_tr = 0;
                            color_tr = color;
                            test_fa += 1;
                        }
                        if (test_tr >= 2)
                        {
                            stop = true;
                        }
                        if (test_fa >= 15)
                        {
                            stop = true;
                        }

                        p_prev = pc;
                        pc += dir;
                        if (!(is_in_img(pc)))
                        {
                            stop = true;
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
                            if (close_point[0].first < 7.0f)
                                pc = points[close_point[0].second];
                            close_point.erase(close_point.begin(), close_point.end());
                            dir = pc - p_prev;
                        }
                        loop += 1;
                    }
                    //std::cout << p_image.get_coord_pix() << " nb loop : " << loop << " color : " << color_tr << std::endl;
                    if (color_tr > NO_COLOR)
                    {
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
                    std::cout << std::endl;
                    std::cout << p_image.get_color_int() << "  " << nghbr.case_color.size() << std::endl;
                    std::cout << " north : " << case_d.north << " sud : " << case_d.south << " east : " << case_d.east << " west : " << case_d.west << std::endl;
                    for (auto t : nghbr.case_color)
                    {
                        std::cout << "color t: " << t.second << std::endl;
                    }
                    if (nghbr.case_color.size() == 2)
                    {
                        // std::cout << p_image.get_color_int() << std::endl;
                        // std::cout << " north : " << case_d.north << " sud : " << case_d.south << " east : " << case_d.east << " west : " << case_d.west << std::endl;
                        int coord_x = 0;
                        int coord_y = 0;
                        // std::cout << "color 1: " << nghbr.case_color[0].second << " color 2: " << nghbr.case_color[1].second << std::endl;
                        // std::cout << "first 1: " << nghbr.case_color[0].first << " first 2: " << nghbr.case_color[1].first << std::endl;
                        switch (c)
                        {

                        case MAGENTA:
                            if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                            {
                                coord_x = 8 - nghbr.case_color[0].first;
                                coord_y = 8 - nghbr.case_color[1].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                            {
                                coord_x = 8 - nghbr.case_color[1].first;
                                coord_y = 8 - nghbr.case_color[0].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            break;
                        case YELLOW:
                            if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                            {
                                coord_x = 8 + nghbr.case_color[1].first;
                                coord_y = 8 + nghbr.case_color[0].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                            {
                                coord_x = 8 + nghbr.case_color[0].first;
                                coord_y = 8 + nghbr.case_color[1].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            break;
                        case CYAN:
                            if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                            {
                                coord_x = 8 - nghbr.case_color[1].first;
                                coord_y = 8 + nghbr.case_color[0].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                            {
                                coord_x = 8 - nghbr.case_color[0].first;
                                coord_y = 8 + nghbr.case_color[1].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            break;
                        case WHITE:
                            if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                            {
                                coord_x = 8 + nghbr.case_color[0].first;
                                coord_y = 8 - nghbr.case_color[1].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                            {
                                coord_x = 8 + nghbr.case_color[1].first;
                                coord_y = 8 - nghbr.case_color[0].first;
                                // std::cout << p_image.get_coord_pix() << " coord x : " << coord_x << " coord y : " << coord_y << std::endl;
                                Point_Mire *p_mire = new Point_Mire(cv::Point2i(coord_x, coord_y), p_image.get_coord_pix(), p_image.get_color_int());
                                p_mire->compute_coords();
                                vector_mire.push_back(p_mire);
                                b++;
                            }
                            break;
                        default:
                            break;
                        }
                    }
                }
            }
        }
    }
    return vector_mire;
}