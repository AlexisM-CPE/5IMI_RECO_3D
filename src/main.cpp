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

    //cv::Mat im_gray_1 = imread("data/origami/1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_1 = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_1 = imread("data/mario/2.jpg", cv::IMREAD_COLOR);

    //cv::Mat im_gray_2 = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_2 = imread("data/mario/7.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_2 = imread("data/mario/7.jpg", cv::IMREAD_COLOR);

    cv::Mat im_BGR_features_1 = im_BGR_1.clone();
    cv::Mat im_BGR_features_2 = im_BGR_2.clone();
    cv::Mat im_features = im_BGR_2.clone();

    cv::Mat im_BGR_2_clone = im_BGR_2.clone();

    // Vectors containing the points used for the calibration
    std::vector<std::vector<cv::Point3f>> object_points_1;
    std::vector<std::vector<cv::Point2f>> image_points_1;

    std::vector<std::vector<cv::Point3f>> object_points_2;
    std::vector<std::vector<cv::Point2f>> image_points_2;

    find_points_mire(im_gray_1, im_BGR_1, object_points_1, image_points_1, "ojb");
    std::cout << "--------IMAGE 4 --------" << std::endl;
    find_points_mire(im_gray_2, im_BGR_2, object_points_2, image_points_2);

    cv::Mat M_int_1(3, 4, CV_64F);
    cv::Mat M_ext_1(4, 4, CV_64F);
    cv::Mat cameraMatrix_1(3, 3, CV_64F);

    cv::Mat M_int_2(3, 4, CV_64F);
    cv::Mat M_ext_2(4, 4, CV_64F);
    cv::Mat cameraMatrix_2(3, 3, CV_64F);

    cv::Mat distCoeffs_1;
    cv::Mat distCoeffs_2;

    Calibrate(im_gray_1, im_BGR_1, object_points_1, image_points_1, cameraMatrix_1, distCoeffs_1, M_int_1, M_ext_1, "lwjfhsdlijfbsdjgbsd");
    Calibrate(im_gray_2, im_BGR_2, object_points_2, image_points_2, cameraMatrix_2, distCoeffs_2, M_int_2, M_ext_2);

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

    // cv::Mat new_image = cv::Mat::zeros(846, 1504, CV_32FC3);
    // for (int i = 0; i < 846; i++)
    // {
    //     for (int j = 0; j < 1504; j++)
    //     {
    //         new_image.at<cv::Vec3f>(i, j) = im_BGR_2.at<cv::Vec3f>(int(i + round(translation_x)), int(j + round(translation_y)));
    //     }
    // }

    // imshow("New image", new_image - im_BGR_1);

    std::vector<cv::Point3f> features_3D = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2);

    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (int i = 0; i < 17; i++)
    {
        for (int j = 0; j < 17; j++)
        {
            cv::Mat p(4, 1, CV_64F);
            p.at<double>(0, 0) = i * 12.375f;
            p.at<double>(1, 0) = j * 12.375f;
            p.at<double>(2, 0) = 0.0f;
            p.at<double>(3, 0) = 1.0f;
            cv::Mat p_m1 = M_transition_1 * p;
            cv::Mat p_m2 = M_transition_2 * p;
            p_m1 /= p_m1.at<double>(2, 0);
            p_m2 /= p_m2.at<double>(2, 0);

            obj.push_back(cv::Point2f(p_m1.at<double>(0, 0) / p_m1.at<double>(2, 0), p_m1.at<double>(1, 0) / p_m1.at<double>(2, 0)));
            scene.push_back(cv::Point2f(p_m2.at<double>(0, 0) / p_m2.at<double>(2, 0), p_m2.at<double>(1, 0) / p_m2.at<double>(2, 0)));
        }
    }

    cv::Mat H = cv::findHomography(obj, scene, cv::RANSAC);

    cv::Mat new_image = cv::Mat::zeros(846, 1504, CV_32FC3);

    cv::Mat H_inv = H.inv();

    std::cout << H_inv.rows << "   " << H_inv.cols << std::endl;

    warpPerspective(im_BGR_1, new_image, H, im_BGR_2.size());

    imshow("New image ", new_image);

    imshow("New image 2", abs(new_image - im_BGR_2));

    cv::Mat im_diff = abs(im_BGR_2 - new_image);
    cv::Mat im_diff_gray;
    cv::cvtColor(im_diff, im_diff_gray, cv::COLOR_BGR2GRAY);
    // cv::Mat im_seg(im_diff.rows, im_diff.cols, CV_32FC3);
    cv::Mat im_seg(846, 1504, CV_32FC3); //
    im_seg = new_image.clone();

    cv::Mat zeros = cv::Mat::zeros(846, 1504, CV_32FC3);

    std::cout << "Type : " << im_seg.type() << std::endl;

    cv::Mat abc = im_diff - im_seg;

    float eps_diff_0 = 30.0f;
    float eps_diff_1 = 0.2f;
    float eps_diff_2 = 0.44f;

    cvtColor(im_BGR_2_clone, im_BGR_2_clone, cv::COLOR_BGR2HSV);
    cvtColor(new_image, new_image, cv::COLOR_BGR2HSV);

    std::cout << new_image.rows << "  " << new_image.cols << std::endl;
    std::cout << im_BGR_2_clone.rows << "  " << im_BGR_2_clone.cols << std::endl;

    for (int i = 0; i < im_diff.rows; ++i)
    {
        for (int j = 0; j < im_diff.cols; ++j)
        {

            if (!((abs(im_BGR_2_clone.at<cv::Vec3f>(i, j)[0] - new_image.at<cv::Vec3f>(i, j)[0]) < eps_diff_0) && (abs(im_BGR_2_clone.at<cv::Vec3f>(i, j)[1] - new_image.at<cv::Vec3f>(i, j)[1]) < eps_diff_1) && (abs(im_BGR_2_clone.at<cv::Vec3f>(i, j)[2] - new_image.at<cv::Vec3f>(i, j)[2]) < eps_diff_2))) // || out_of_rectangle(i, j, M_transition_2))
            {
                //std::cout << im_seg.at<cv::Vec3f>(i, j).type() << std::endl;
                im_seg.at<cv::Vec3f>(i, j) = zeros.at<cv::Vec3f>(i, j); //cv::Vec3f(0, 0, 0);
                // std::cout << im_diff_gray.at<float>(i, j) << std::endl;
            }
            else
            {
                im_seg.at<cv::Vec3f>(i, j) = im_BGR_2.at<cv::Vec3f>(i, j);
            }
            std::cout << i << " " << j << std::endl;
        }
    }
    imshow("Gray ", im_diff_gray);
    imshow("New diff ", im_seg);
    imshow("diff ", im_diff);

    std::vector<cv::Point2f> matched_transformed_1;
    std::vector<cv::Point2f> matched_transformed_2;

    for (int i = 0; i < matched_points1.size(); i++)
    {
        circle(im_BGR_features_1, matched_points1[i], 1, cv::Scalar(0, 255, 0), 2);
        circle(im_BGR_features_2, matched_points2[i], 1, cv::Scalar(0, 0, 255), 2);
    }

    imshow("Features image 1", im_BGR_features_1);
    imshow("Features image 2", im_BGR_features_2);

    float eps = 100.0f;
    for (int i = 0; i < matched_points1.size(); i++)
    {
        cv::Mat p(3, 1, CV_64F);
        p.at<double>(0, 0) = matched_points1[i].x;
        p.at<double>(1, 0) = matched_points1[i].y;
        p.at<double>(2, 0) = 1.0f;

        cv::Mat p_m = H * p;

        cv::Point2f p_transformed(p_m.at<double>(0, 0) / p_m.at<double>(2, 0), p_m.at<double>(1, 0) / p_m.at<double>(2, 0));

        float a = sqrt(pow(p_transformed.x - matched_points2[i].x, 2) + pow(p_transformed.y - matched_points2[i].y, 2));

        if (a < eps)
        {
            matched_transformed_1.push_back(matched_points2[i]);
            matched_transformed_2.push_back(p_transformed);
        }
    }

    std::cout << matched_transformed_1.size() << std::endl;

    for (int i = 0; i < matched_transformed_1.size(); i++)
    {
        circle(im_features, matched_transformed_1[i], 1, cv::Scalar(0, 255, 0), 2);
        circle(im_features, matched_transformed_2[i], 1, cv::Scalar(0, 0, 255), 2);
    }

    imshow("Features", im_features);

    while (true)
    {
        // Close and quit only when Escape is pressed
        int key = cv::waitKey(0);
        if (key == 27 || key == -1)
            break;
    }
    return 0;
}