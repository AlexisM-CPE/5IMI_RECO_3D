/* features_extraction.hpp */
#ifndef FEATURES_EXTRACTION_HPP
#define FEATURES_EXTRACTION_HPP
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

void extract_features(cv::Mat image_in1, cv::Mat image_in2, cv::Mat *image_out1, cv::Mat *image_out2, int threshold);

#endif
