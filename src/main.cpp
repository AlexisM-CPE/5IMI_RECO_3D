#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include "segmentation.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "feature_location.hpp"

#include "utils.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
    // Loads an image
    cv::Mat im_gray_1 = imread("data/origami/1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_1 = imread("data/origami/1.jpg", cv::IMREAD_COLOR);

    cv::Mat im_gray_2 = imread("data/origami/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_2 = imread("data/origami/2.jpg", cv::IMREAD_COLOR);

    // Vectors containing the points used for the calibration
    std::vector<std::vector<cv::Point3f>> object_points_1;
    std::vector<std::vector<cv::Point2f>> image_points_1;

    std::vector<std::vector<cv::Point3f>> object_points_2;
    std::vector<std::vector<cv::Point2f>> image_points_2;

    find_points_mire(im_gray_1, im_BGR_1, object_points_1, image_points_1, "Image 1");
    find_points_mire(im_gray_2, im_BGR_2, object_points_2, image_points_2, "Image 2");

    cv::Mat M_int_1(3, 4, CV_64F);
    cv::Mat M_ext_1(4, 4, CV_64F);
    cv::Mat cameraMatrix_1(3, 3, CV_64F);

    cv::Mat M_int_2(3, 4, CV_64F);
    cv::Mat M_ext_2(4, 4, CV_64F);
    cv::Mat cameraMatrix_2(3, 3, CV_64F);

    Calibrate(im_gray_1, im_BGR_1, object_points_1, image_points_1, cameraMatrix_1, M_int_1, M_ext_1, "Calibrage image 1");
    Calibrate(im_gray_2, im_BGR_2, object_points_2, image_points_2, cameraMatrix_2, M_int_2, M_ext_2, "Calibrage image 2");

    // Segmentation
    cv::Mat segmented;
    cv::Mat M_trans_seg = compute_transition_matrix(M_int_1, M_ext_1);
    segmentation(im_BGR_1, M_trans_seg, segmented);
    imshow("segmentation", segmented);


    cv::Mat imageo1, imageo2;
    std::vector<cv::Point2f> matched_points1;
    std::vector<cv::Point2f> matched_points2;
    extract_features(im_gray_1, im_gray_2, &imageo1, &imageo2, &matched_points1, &matched_points2, 100);

    cv::Point3f camera_pos_1 = get_camera_position(M_ext_1);
    cv::Point3f camera_pos_2 = get_camera_position(M_ext_2);


    cv::Mat M_transition_1 = compute_transition_matrix(M_int_1, M_ext_1);
    cv::Mat M_transition_2 = compute_transition_matrix(M_int_2, M_ext_2);

    std::vector<cv::Point3f> features_3D = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2);
    


    create_cloud_file(features_3D, "./nuage.xyz");
    
    for (auto p : matched_points1)
    {
        circle(im_gray_1, p, 15 / 2, cv::Scalar(255, 0, 0), 1);
    }

    imshow("features", im_gray_1);

    while (true)
    {
        // Close and quit only when Escape is pressed
        int key = cv::waitKey(0);
        if (key == 27 || key == -1)
            break;
    }
    return 0;
}