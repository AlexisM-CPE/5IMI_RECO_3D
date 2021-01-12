#include "segmentation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


void segmentation(const cv::Mat &im_to_segment, const cv::Mat &image_to_object, cv::Mat &segmented){
    segmented = cv::Mat(cv::Size(im_to_segment.rows, im_to_segment.cols), CV_64FC3);
    cv::Mat grid = cv::imread("data/grid/grid.png", cv::IMREAD_COLOR);
    grid.convertTo(grid, CV_32FC3, 1.0 / 255.0);

}
