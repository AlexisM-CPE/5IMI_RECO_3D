#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(int argc, const char* argv[])
{
    Mat image;
    Mat HSV_image;

    Mat magenta(1,1,CV_32FC3);
    Mat HSV_magenta;
    magenta = cv::Scalar(1.0f, 0, 1.0f); 
    cvtColor(magenta, HSV_magenta, COLOR_BGR2HSV);

    Mat yellow(1,1,CV_32FC3);
    Mat HSV_yellow;
    yellow = cv::Scalar(0.0f, 1.0f, 1.0f); 
    cvtColor(yellow, HSV_yellow, COLOR_BGR2HSV);

    Mat cyan(1,1,CV_32FC3);
    Mat HSV_cyan;
    cyan = cv::Scalar(1.0f, 0, 1.0f); 
    cvtColor(cyan, HSV_cyan, COLOR_BGR2HSV);

    image = imread( "bougie/20201218_170938.jpg", 1 );
    image.convertTo(image,CV_32FC3,  1.0/255.0);
    cvtColor(image, HSV_image, COLOR_BGR2HSV);

    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    
    //-- Show detected (drawn) keypoints
    imshow("Image", HSV_image );
    resizeWindow("Image", image.cols, image.rows);

    std::cout<<std::hex << HSV_image.at<cv::Vec3f>(4,3)[0] << std::endl;
    std::cout<<std::hex << type2str(image.type()) << std::endl;
    waitKey(0);
    return 0;
}