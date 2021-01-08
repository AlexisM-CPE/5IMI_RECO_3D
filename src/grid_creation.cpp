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
    Vec3f hsv_color = HSV.at<Vec3f>(p);

    int i = 0;
    float H_tot = 0.0f;
    float S_tot = 0.0f;

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
                    i++;
                }
            }
        }
    }
    float H = H_tot / i;
    float S = S_tot / i;
    if (i < 150)
    {
        return NO_COLOR;
    }
    else if (S < 0.4f)
    {
        if (S < 0.1f)
            return WHITE;
        else
            return NO_COLOR;
    }
    else if (abs(HSV_magenta.at<Vec3f>(0, 0)[0] - H) < 40.0f)
    {
        return MAGENTA;
    }
    else if (abs(HSV_yellow.at<Vec3f>(0, 0)[0] - H) < 40.0f)
    {
        return YELLOW;
    }
    else if (abs(HSV_cyan.at<Vec3f>(0, 0)[0] - H) < 40.0f)
    {
        return CYAN;
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

void find_pos(Mat HSV, vector<Point2f> points)
{
    for (int i = 0; i < points.size(); i++)
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
            if (abs(close_norm[0].first - close_norm[3].first) > 7.0f)
                verif = false;

            if (!((find_color(HSV, points[close_norm[0].second]) == c) && (find_color(HSV, points[close_norm[1].second]) == c) && (find_color(HSV, points[close_norm[2].second]) == c) && (find_color(HSV, points[close_norm[3].second]) == c)))
                verif = false;
            if (verif)
            {
                neighbour nghbr;
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
                    while (stop == false)
                    {
                        int color = find_color(HSV, pc);
                        if ((color != c) && (false_find == false))
                        {
                            false_find = true;
                            dir_name = find_dir(dir, case_d, loop);
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
                //std::cout << " north : " << case_d.north << " sud : " << case_d.south << " east : " << case_d.east << " west : " << case_d.west << std::endl;

                if (nghbr.case_color.size() == 2)
                {
                    switch (c)
                    {
                    case MAGENTA:
                        if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                        {
                            //std::cout << points[i] << " coord x : " << 8 - nghbr.case_color[0].first << " coord y : " << 8 - nghbr.case_color[1].first << std::endl;
                        }
                        else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                        {
                            //std::cout << points[i] << " coord x : " << 8 - nghbr.case_color[1].first << " coord y : " << 8 - nghbr.case_color[0].first << std::endl;
                        }
                        break;
                    case YELLOW:
                        if ((nghbr.case_color[0].second == WHITE) && (nghbr.case_color[1].second == CYAN))
                        {
                            // std::cout << points[i] << " coord x : " << 8 + nghbr.case_color[1].first << " coord y : " << 8 + nghbr.case_color[0].first << std::endl;
                        }
                        else if ((nghbr.case_color[1].second == WHITE) && (nghbr.case_color[0].second == CYAN))
                        {
                            // std::cout << points[i] << " coord x : " << 8 + nghbr.case_color[0].first << " coord y : " << 8 + nghbr.case_color[1].first << std::endl;
                        }
                        break;
                    case CYAN:
                        if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                        {
                            // std::cout << points[i] << " coord x : " << 8 - nghbr.case_color[1].first << " coord y : " << 8 + nghbr.case_color[0].first << std::endl;
                        }
                        else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                        {
                            // std::cout << points[i] << " coord x : " << 8 + nghbr.case_color[0].first << " coord y : " << 8 - nghbr.case_color[1].first << std::endl;
                        }
                        break;
                    case WHITE:
                        if ((nghbr.case_color[0].second == MAGENTA) && (nghbr.case_color[1].second == YELLOW))
                        {
                            // std::cout << points[i] << " coord x : " << 8 + nghbr.case_color[0].first << " coord y : " << 8 - nghbr.case_color[1].first << std::endl;
                        }
                        else if ((nghbr.case_color[1].second == MAGENTA) && (nghbr.case_color[0].second == YELLOW))
                        {
                            // std::cout << points[i] << " coord x : " << 8 + nghbr.case_color[1].first << " coord y : " << 8 - nghbr.case_color[0].first << std::endl;
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