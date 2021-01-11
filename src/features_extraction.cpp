/* features_extraction.cpp */
#include "features_extraction.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/xfeatures2d.hpp"

std::vector<cv::KeyPoint> extract_features(cv::Mat image, int threshold)
{
    std::vector<cv::KeyPoint> keypointsD;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    std::vector<cv::Mat> descriptor;
    detector->detect(image, keypointsD, cv::Mat());
    cv::KeyPointsFilter::retainBest(keypointsD, threshold);

    return keypointsD;
}