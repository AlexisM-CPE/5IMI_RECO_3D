#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"


void segmentation(const cv::Mat &im_to_segment, const cv::Mat &image_to_object, cv::Mat &segmented);


#endif