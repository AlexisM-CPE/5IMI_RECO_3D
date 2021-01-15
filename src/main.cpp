#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include "segmentation.hpp"
#include "registration.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "feature_location.hpp"

#include "utils.hpp"
#include <iostream>
#include <cmath>

int main(int argc, char **argv)
{
    // Loads an image

    //cv::Mat im_gray_1 = imread("data/origami/1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_1 = imread("data/mario/18.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_1 = imread("data/mario/18.jpg", cv::IMREAD_COLOR);

    //cv::Mat im_gray_2 = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_2 = imread("data/mario/19.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_2 = imread("data/mario/19.jpg", cv::IMREAD_COLOR);

    // Vectors containing the points used for the calibration
    std::vector<std::vector<cv::Point3f>> object_points_1;
    std::vector<std::vector<cv::Point2f>> image_points_1;

    std::vector<std::vector<cv::Point3f>> object_points_2;
    std::vector<std::vector<cv::Point2f>> image_points_2;

    find_points_mire(im_gray_1, im_BGR_1, object_points_1, image_points_1, "Image 1");
    std::cout << "--------IMAGE 4 --------" << std::endl;
    find_points_mire(im_gray_2, im_BGR_2, object_points_2, image_points_2, "Image 4");

    cv::Mat M_int_1(3, 4, CV_64F);
    cv::Mat M_ext_1(4, 4, CV_64F);
    cv::Mat cameraMatrix_1(3, 3, CV_64F);

    cv::Mat M_int_2(3, 4, CV_64F);
    cv::Mat M_ext_2(4, 4, CV_64F);
    cv::Mat cameraMatrix_2(3, 3, CV_64F);

    cv::Mat distCoeffs_1;
    cv::Mat distCoeffs_2;

    Calibrate(im_gray_1, im_BGR_1, object_points_1, image_points_1, cameraMatrix_1, distCoeffs_1, M_int_1, M_ext_1, "Calibrage image 1");
    Calibrate(im_gray_2, im_BGR_2, object_points_2, image_points_2, cameraMatrix_2, distCoeffs_2, M_int_2, M_ext_2, "Calibrage image 4");

    // Segmentation
    cv::Mat segmented;
    cv::Mat M_trans_seg = compute_transition_matrix(M_int_1, M_ext_1);
    //segmentation(im_BGR_1, M_trans_seg, segmented);
    //imshow("segmentation", segmented);

    cv::Mat imageo1, imageo2;
    std::vector<cv::Point2f> matched_points1;
    std::vector<cv::Point2f> matched_points2;
    extract_features(im_gray_1, im_gray_2, &imageo1, &imageo2, &matched_points1, &matched_points2, 10000);

    cv::Point3f camera_pos_1 = get_camera_position(M_ext_1);
    cv::Point3f camera_pos_2 = get_camera_position(M_ext_2);

    std::cout << " Cam  : " << camera_pos_2.x << "   " << camera_pos_2.y << "   " << camera_pos_2.z << std::endl;

    cv::Mat M_transition_1 = compute_transition_matrix(M_int_1, M_ext_1);
    cv::Mat M_transition_2 = compute_transition_matrix(M_int_2, M_ext_2);

    cv::Mat centre_mire(4, 1, CV_64F);
    centre_mire.at<double>(0, 0) = 8 * 12.375f;
    centre_mire.at<double>(1, 0) = 8 * 12.375f;
    centre_mire.at<double>(2, 0) = 0.0f;
    centre_mire.at<double>(3, 0) = 1.0f;

    cv::Mat centre_mire_im_1 = M_transition_1 * centre_mire;
    cv::Mat centre_mire_im_2 = M_transition_2 * centre_mire;

    centre_mire_im_1 /= centre_mire_im_1.at<double>(2, 0);
    centre_mire_im_2 /= centre_mire_im_2.at<double>(2, 0);

    float translation_y = centre_mire_im_2.at<double>(0, 0) - centre_mire_im_1.at<double>(0, 0);
    float translation_x = centre_mire_im_2.at<double>(1, 0) - centre_mire_im_1.at<double>(1, 0);

    cv::Mat new_image = cv::Mat::zeros(846, 1504, CV_32FC3);
    for (int i = 0; i < 846; i++)
    {
        for (int j = 0; j < 1504; j++)
        {
            new_image.at<cv::Vec3f>(i, j) = im_BGR_2.at<cv::Vec3f>(int(i + round(translation_x)), int(j + round(translation_y)));
        }
    }

    imshow("New image", new_image - im_BGR_1);

    std::vector<cv::Point3f> features_3D = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2);

    create_cloud_file(features_3D, "./nuage.xyz");

    for (auto p : matched_points1)
    {
        circle(im_gray_1, p, 2 / 2, cv::Scalar(255, 0, 0), 1);
    }

    cv::Point3f c1(0.0f, 0.0f, 0.0f);
    cv::Point3f c2(16.0f, 0.0f, 0.0f);
    cv::Point3f c3(0.0f, 16.0f, 0.0f);
    cv::Point3f c4(16.0f, 16.0f, 0.0f);

    std::vector<cv::Point3f> coins;
    coins.push_back(c1);
    coins.push_back(c2);
    coins.push_back(c3);
    coins.push_back(c4);

    //imshow("features", im_gray_1);

    TransformType::Pointer transform = TransformType::New();
    transform = registrate_image("data/mario/3.jpg", "data/mario/4.jpg");
    cv::Mat im_out = imread("out/output.jpg", cv::IMREAD_GRAYSCALE);

    cv::Mat diff = im_gray_2 - im_out;
    imshow("diff", diff);

    diff = im_gray_1 - im_out;
    imshow("diff2", diff);
    while (true)
    {
        // Close and quit only when Escape is pressed
        int key = cv::waitKey(0);
        if (key == 27 || key == -1)
            break;
    }
    return 0;
}