/* features_extraction.hpp */
#ifndef FEATURES_EXTRACTION_HPP
#define FEATURES_EXTRACTION_HPP
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

std::vector<cv::KeyPoint> extract_features(cv::Mat image, int threshold);

#endif
