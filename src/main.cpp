#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "registration.hpp"
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

    cv::Mat imageo1, imageo2;
    std::vector<cv::Point2f> matched_points1;
    std::vector<cv::Point2f> matched_points2;
    extract_features(im_gray_1, im_gray_2, &imageo1, &imageo2, &matched_points1, &matched_points2, 1000);

    cv::Point3f camera_pos_1 = get_camera_position(M_ext_1);
    cv::Point3f camera_pos_2 = get_camera_position(M_ext_2);

    cv::Mat M_transition_1 = compute_transition_matrix(M_int_1, M_ext_1);
    cv::Mat M_transition_2 = compute_transition_matrix(M_int_2, M_ext_2);

    std::vector<cv::Point3f> features_3D = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2);

    create_cloud_file(features_3D, "./nuage.xyz");

    TransformType::Pointer transform = TransformType::New();
    transform = registrate_image("data/origami/2.jpg", "data/origami/1.jpg");

    std::vector<cv::Point2f> new_matched_points2;
    for (auto p : matched_points1)
    {
        PointType temp = transform_point(convert_CVPoint2ITKPoint(p), transform);
        cv::Point2f p_temp = convert_ITKPoint2CVPoint(temp);
        new_matched_points2.push_back(p_temp);
        circle(im_BGR_2, p_temp, 2, cv::Scalar(0, 0, 255), -1);
    }

    for (auto p : matched_points1)
    {
        circle(im_BGR_2, p, 2, cv::Scalar(255, 0, 0), -1);
    }
    std::cout << "old x : " << matched_points2[0].x << " y : " << matched_points2[0].y << std::endl;
    std::cout << "new x : " << new_matched_points2[0].x << " y : " << new_matched_points2[0].y << std::endl;
    imshow("features", im_BGR_2);

    cv::Mat im_out = imread("out/output.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat diff = im_gray_2 - im_out;
    imshow("diff", diff);
    while (true)
    {
        // Close and quit only when Escape is pressed
        int key = cv::waitKey(0);
        if (key == 27 || key == -1)
            break;
    }
    return 0;
}