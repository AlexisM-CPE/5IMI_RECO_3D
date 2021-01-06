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
            if(HSV.at<Vec3f>(p - Point2f(j,k))[2] > 0.44f)
            {
                H_tot += HSV.at<Vec3f>(p - Point2f(j,k))[0];
                S_tot += HSV.at<Vec3f>(p - Point2f(j,k))[1];
                i++;
            } 
            
        }
    }
    float H = H_tot/i;
    float S = S_tot/i;
    //cout<<"H : "<<H<<" S : "<<S<<" i : "<<i<<endl;

    if(S < 0.1f)
    {
        return WHITE;
    }
    if(abs(HSV_magenta.at<Vec3f>(0,0)[0]-H) < 30.0f)
    {
        return MAGENTA;
    }
    else if(abs(HSV_yellow.at<Vec3f>(0,0)[0]-H) < 30.0f)
    {
        return YELLOW;
    }
    else if(abs(HSV_cyan.at<Vec3f>(0,0)[0]-H) < 30.0f)
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
    //cout<<"color : "<<c<<endl;
    vector<pair<float, int>> close_norm;
    int j = 0;
    for(auto p : points)
    {
      close_norm.push_back(make_pair(norm(p-points[i]),j));
      j++;
    }
    sort(close_norm.begin(), close_norm.end());
    close_norm.erase(close_norm.begin());

    //cout<<close_norm[0].first<<" "<<close_norm[1].first<<" "<<close_norm[2].first<<" "<<close_norm[3].first<<" "<<endl;
  }
}