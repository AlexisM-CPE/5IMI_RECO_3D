/* features_extraction.cpp */
#include "features_extraction.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>

void extract_features(cv::Mat image_in1, cv::Mat image_in2, cv::Mat *image_out1, cv::Mat *image_out2, std::vector<cv::Point2f> *matched_points1, std::vector<cv::Point2f> *matched_points2, int threshold)
{
    // Feature detection
    std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::ORB> detector = cv::ORB::create(threshold);
    detector->detectAndCompute(image_in1, cv::Mat(), keyPoints1, descriptors1);
    detector->detectAndCompute(image_in2, cv::Mat(), keyPoints2, descriptors2);

    // Feature matching
    cv::BFMatcher BF = cv::BFMatcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> matches;
    BF.knnMatch(descriptors1, descriptors2, matches, 2);

    std::vector<cv::DMatch> match1;
    std::vector<cv::DMatch> match2;
    std::vector<cv::Point2f> test1;

    for (int i = 0; i < matches.size(); i++)
    {
        match1.push_back(matches[i][0]);
        match2.push_back(matches[i][1]);
        matched_points1->push_back(keyPoints1[matches[i][0].queryIdx].pt);
        matched_points2->push_back(keyPoints2[matches[i][1].queryIdx].pt);
    }
    cv::drawMatches(image_in1, keyPoints1, image_in2, keyPoints2, match1, *image_out1);
    cv::drawMatches(image_in1, keyPoints1, image_in2, keyPoints2, match2, *image_out2);
    //cv::KeyPointsFilter::retainBest(keypointsD, threshold);
}