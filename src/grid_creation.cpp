#include "grid_creation.hpp"

using namespace cv;
using namespace std;

int find_color(Mat HSV, Point2f p)
{
    Mat magenta(1,1,CV_32FC3);
    Mat HSV_magenta;
    magenta = Scalar(1.0f, 0, 1.0f); 
    cvtColor(magenta, HSV_magenta, COLOR_BGR2HSV);

    Mat yellow(1,1,CV_32FC3);
    Mat HSV_yellow;
    yellow = Scalar(0.0f, 1.0f, 1.0f); 
    cvtColor(yellow, HSV_yellow, COLOR_BGR2HSV);

    Mat cyan(1,1,CV_32FC3);
    Mat HSV_cyan;
    cyan = Scalar(1.0f, 1.0f, 0.0f); 
    cvtColor(cyan, HSV_cyan, COLOR_BGR2HSV);
    Vec3f hsv_color = HSV.at<Vec3f>(p);

    int i = 0;
    float H_tot = 0.0f;
    float S_tot = 0.0f;

    for(float j = -10; j < 10; j++)
    {
        for(float k = -10; k < 10; k++)
        {
            Point2f p_temp = p - Point2f(j,k);
            if(!((p_temp.x < 0)||(p_temp.y < 0)||(p_temp.y >= 846)||(p_temp.x >= 1504)))
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
    float H = H_tot/i;
    float S = S_tot/i;
    if(i < 150)
    {
        return NO_COLOR;
    }
    else if(S < 0.4f)
    {
        if(S < 0.1f) 
            return WHITE;
        else
            return NO_COLOR;
    }
    else if(abs(HSV_magenta.at<Vec3f>(0,0)[0]-H) < 40.0f)
    {
        return MAGENTA;
    }
    else if(abs(HSV_yellow.at<Vec3f>(0,0)[0]-H) < 40.0f)
    {
        return YELLOW;
    }
    else if(abs(HSV_cyan.at<Vec3f>(0,0)[0]-H) < 40.0f)
    {
        return CYAN;
    }
    else
    {
        return NO_COLOR;
    }
} 

void find_pos(Mat HSV, vector<Point2f> points)
{
    for(int i = 0; i < points.size(); i++)
    {
        int c = find_color(HSV, points[i]);
        if(c != NO_COLOR)
        {
            vector<pair<float, int>> close_norm;
            int j = 0;
            for (auto p : points)
            {
                close_norm.push_back(make_pair(norm(p - points[i]), j));
                j++;
            }
            sort(close_norm.begin(), close_norm.end());
            close_norm.erase(close_norm.begin());
            close_norm.erase(close_norm.begin() + 4, close_norm.end());
            bool verif = true;
            for (int k = 0; k < 3; k++)
            {
                for(int l = 0; l < 4; l++)
                {
                    if (abs(close_norm[k].first - close_norm[k].first) > 7.0f)
                        verif = false;
                    if (find_color(HSV, points[close_norm[k].second]) != c)
                        verif = false;
                }
            }
            if(verif)
            {
                for (int k = 0; k < 4; k++)
                {
                    Point2f dir = points[close_norm[k].second] - points[i];
                    int loop = 0;
                    bool stop = false;
                    int test_fa = 0;
                    int test_tr = 0;
                    int color_tr = 0;
                    Point2f pc = points[i] + dir;
                    Point2f p_prev = points[i];
                    vector<pair<float, int>> close_point;
                    while(stop == false)
                    {
                        int color = find_color(HSV, pc);
                        if (color == NO_COLOR)
                        {
                            test_fa += 1;
                        }
                        else if (color == color_tr)
                        {
                            test_tr += 1;
                            test_fa = 0;
                        }
                        else if(color != c)
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
                        loop += 1;
                        p_prev = pc;
                        pc += dir;
                        if ((pc.x >= 1504) || (pc.x < 0))
                        {
                            stop = true;
                        }
                        if ((pc.y >= 846) || (pc.y < 0))
                        {
                            stop = true;
                        }
                        if(stop == false)
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
                    }
                    std::cout << points[i] << " nb loop : " << loop << " color : " << color_tr << std::endl;
                }
            }
        }
    }
}