/* features_extraction.cpp */
#include "features_extraction.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"

#include <iostream>

#define dist_eps 2.0f

void extract_features(cv::Mat image_in1, cv::Mat image_in2, cv::Mat *image_out1, cv::Mat *image_out2, std::vector<cv::Point2f> *features_finale_1, std::vector<cv::Point2f> *features_finale_2, int threshold)
{

    //Feature detection
    // std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    // cv::Mat descriptors1, descriptors2;
    // cv::Ptr<cv::ORB> detector = cv::ORB::create(threshold, 1.2, 2, 31, 0, 4, cv::ORB::HARRIS_SCORE, 31);
    // detector->detectAndCompute(image_in1, cv::Mat(), keyPoints1, descriptors1);
    // detector->detectAndCompute(image_in2, cv::Mat(), keyPoints2, descriptors2);
    std::vector<cv::Point2f> *matched_points1 = new std::vector<cv::Point2f>();
    std::vector<cv::Point2f> *matched_points2 = new std::vector<cv::Point2f>();
    std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    cv::Mat descriptors1, descriptors2;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(10, 4, 3, false, false);
    detector->detectAndCompute(image_in1, cv::Mat(), keyPoints1, descriptors1);
    detector->detectAndCompute(image_in2, cv::Mat(), keyPoints2, descriptors2);

    // std::cout << keyPoints1.size() << "   " << keyPoints2.size() << std::endl;

    // std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
    // cv::Mat descriptors1, descriptors2;
    // cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();
    // detector->detect(image_in1, keyPoints1);
    // detector->detect(image_in2, keyPoints2);
    // detector->compute(image_in1, keyPoints1, descriptors1);
    // detector->compute(image_in2, keyPoints2, descriptors2);

    descriptors1.convertTo(descriptors1, CV_32F);
    descriptors2.convertTo(descriptors2, CV_32F);
    // Feature matching
    // cv::BFMatcher BF = cv::BFMatcher(cv::NORM_HAMMING);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> matches;
    // BF.knnMatch(descriptors1, descriptors2, matches, 2);
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    // std::cout << matches.size() << std::endl;

    std::vector<cv::DMatch> match1;
    std::vector<cv::DMatch> match2;
    std::vector<cv::Point2f> test1;

    const float ratio_thresh = 1.0f;
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
        {
            match1.push_back(matches[i][0]);
            match2.push_back(matches[i][1]);
            matched_points1->push_back(keyPoints1[matches[i][0].queryIdx].pt);
            matched_points2->push_back(keyPoints2[matches[i][0].trainIdx].pt);
        }
    }

    cv::Mat H_features_2_to_1 = cv::findHomography(*matched_points2, *matched_points1, cv::RANSAC);
    H_features_2_to_1.convertTo(H_features_2_to_1, CV_32F, 1.0f);
    std::vector<cv::Point2f> features_1_in_1;
    std::vector<cv::Point2f> features_2_in_1;
    std::vector<cv::Point2f> features_2_in_2;

    for (int i = 0; i < keyPoints1.size(); i++)
    {
        features_1_in_1.push_back(keyPoints1[i].pt);
    }

    for (int i = 0; i < keyPoints2.size(); i++)
    {
        features_2_in_2.push_back(keyPoints2[i].pt);
    }

    for (int i = 0; i < features_2_in_2.size(); i++)
    {
        cv::Point2f p_features(features_2_in_2[i]);

        cv::Mat p_in_2(3, 1, CV_32F);
        p_in_2.at<float>(0, 0) = p_features.x;
        p_in_2.at<float>(1, 0) = p_features.y;
        p_in_2.at<float>(2, 0) = 1.0f;

        cv::Mat p_in_1(3, 1, CV_32F);
        p_in_1 = H_features_2_to_1 * p_in_2;
        cv::Point2f p(p_in_1.at<float>(0, 0) / p_in_1.at<float>(2, 0), p_in_1.at<float>(1, 0) / p_in_1.at<float>(2, 0));

        features_2_in_1.push_back(p);
    }

    for (int i = 0; i < features_1_in_1.size(); i++)
    {
        float dist_min = 10000.0f;
        int index = -1;
        for (int j = 0; j < features_2_in_1.size(); j++)
        {
            float dist = sqrt(pow(features_1_in_1[i].x - features_2_in_1[j].x, 2) + pow(features_1_in_1[i].y - features_2_in_1[j].y, 2));
            if (dist < dist_min)
            {
                dist_min = dist;
                index = j;
            }
        }
        if (dist_min < dist_eps)
        {
            features_finale_1->push_back(features_1_in_1[i]);
            features_finale_2->push_back(features_2_in_2[index]);
        }
    }

    for (int i = 0; i < features_finale_1->size(); i++)
    {
        circle(image_in1, (*features_finale_1)[i], 1, cv::Scalar(255, 0, 0), 2);
    }
    imshow("Features 2 to 1", image_in1);
    // cv::Mat image_out(846, 1504, CV_32FC3);

    // image_in1.convertTo(image_in1, CV_32FC3, 1.0f / 255.0f);
    // image_in2.convertTo(image_in2, CV_32FC3, 1.0f / 255.0f);

    // std::cout << image_in2.type() << " " << image_in1.type() << " " << image_out.type() << " " << H_features_2_to_1.type() << std::endl;
    // warpPerspective(image_in2, image_out, H_features_2_to_1, image_in1.size());

    // image_out.convertTo(image_out, CV_8UC3, 255.0f);
    // image_in1.convertTo(image_in1, CV_8UC3, 255.0f);
    // image_in2.convertTo(image_in2, CV_8UC3, 255.0f);

    // imshow("Warp features", image_out);

    cv::drawMatches(image_in1, keyPoints1, image_in2, keyPoints2, match1, *image_out1, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::drawMatches(image_in1, keyPoints1, image_in2, keyPoints2, match2, *image_out2, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;
    for (size_t i = 0; i < match1.size(); i++)
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keyPoints1[match1[i].queryIdx].pt);
        scene.push_back(keyPoints2[match1[i].trainIdx].pt);
    }

    cv::Mat im1 = image_in1.clone();
    cv::drawKeypoints(im1, keyPoints1, im1);

    imshow("Keypoints image 1", im1);
    // cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);
    // //-- Get the corners from the image_1 ( the object to be "detected" )
    // std::vector<cv::Point2f> obj_corners(4);
    // obj_corners[0] = cv::Point2f(0, 0);
    // obj_corners[1] = cv::Point2f((float)image_in2.cols, 0);
    // obj_corners[2] = cv::Point2f((float)image_in2.cols, (float)image_in2.rows);
    // obj_corners[3] = cv::Point2f(0, (float)image_in2.rows);
    // std::vector<cv::Point2f> scene_corners(4);

    // cv::perspectiveTransform(obj_corners, scene_corners, H);
    // //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    // line(*image_out1, scene_corners[0] + cv::Point2f((float)image_in2.cols, 0),
    //      scene_corners[1] + cv::Point2f((float)image_in2.cols, 0), cv::Scalar(0, 255, 0), 4);
    // line(*image_out1, scene_corners[1] + cv::Point2f((float)image_in2.cols, 0),
    //      scene_corners[2] + cv::Point2f((float)image_in2.cols, 0), cv::Scalar(0, 255, 0), 4);
    // line(*image_out1, scene_corners[2] + cv::Point2f((float)image_in2.cols, 0),
    //      scene_corners[3] + cv::Point2f((float)image_in2.cols, 0), cv::Scalar(0, 255, 0), 4);
    // line(*image_out1, scene_corners[3] + cv::Point2f((float)image_in2.cols, 0),
    //      scene_corners[0] + cv::Point2f((float)image_in2.cols, 0), cv::Scalar(0, 255, 0), 4);

    cv::imshow("1", *image_out1);
    //cv::KeyPointsFilter::retainBest(keypointsD, threshold);
}