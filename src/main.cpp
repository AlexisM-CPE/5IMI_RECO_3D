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

std::string type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}
int main(int argc, char **argv)
{
    // Loads an image

    //cv::Mat im_gray_1 = imread("data/origami/1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_mire = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_mire = imread("data/mario/2.jpg", cv::IMREAD_COLOR);

    cv::Mat im_gray_1 = imread("data/mario/6.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_1 = imread("data/mario/6.jpg", cv::IMREAD_COLOR);

    //cv::Mat im_gray_2 = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_2 = imread("data/mario/7.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_2 = imread("data/mario/7.jpg", cv::IMREAD_COLOR);

    cv::Mat im_BGR_features_1 = im_BGR_1.clone();
    cv::Mat im_BGR_features_2 = im_BGR_2.clone();
    cv::Mat im_features = im_BGR_2.clone();

    cv::Mat im_BGR_2_clean = im_BGR_2.clone();
    cv::Mat im_BGR_1_clean = im_BGR_1.clone();
    cv::Mat im_BGR_mire_clean = im_BGR_mire.clone();

    // Vectors containing the points used for the calibration

    std::vector<std::vector<cv::Point3f>> object_points_mire;
    std::vector<std::vector<cv::Point2f>> image_points_mire;

    std::vector<std::vector<cv::Point3f>> object_points_1;
    std::vector<std::vector<cv::Point2f>> image_points_1;

    std::vector<std::vector<cv::Point3f>> object_points_2;
    std::vector<std::vector<cv::Point2f>> image_points_2;

    find_points_mire(im_gray_mire, im_BGR_mire, object_points_mire, image_points_mire);
    find_points_mire(im_gray_1, im_BGR_1, object_points_1, image_points_1);
    find_points_mire(im_gray_2, im_BGR_2, object_points_2, image_points_2);

    cv::Mat M_int_mire(3, 4, CV_64F);
    cv::Mat M_ext_mire(4, 4, CV_64F);
    cv::Mat cameraMatrix_mire(3, 3, CV_64F);

    cv::Mat M_int_1(3, 4, CV_64F);
    cv::Mat M_ext_1(4, 4, CV_64F);
    cv::Mat cameraMatrix_1(3, 3, CV_64F);

    cv::Mat M_int_2(3, 4, CV_64F);
    cv::Mat M_ext_2(4, 4, CV_64F);
    cv::Mat cameraMatrix_2(3, 3, CV_64F);

    cv::Mat distCoeffs_mire;
    cv::Mat distCoeffs_1;
    cv::Mat distCoeffs_2;

    Calibrate(im_gray_mire, im_BGR_mire, object_points_mire, image_points_mire, cameraMatrix_mire, distCoeffs_mire, M_int_mire, M_ext_mire);
    Calibrate(im_gray_1, im_BGR_1, object_points_1, image_points_1, cameraMatrix_1, distCoeffs_1, M_int_1, M_ext_1);
    Calibrate(im_gray_2, im_BGR_2, object_points_2, image_points_2, cameraMatrix_2, distCoeffs_2, M_int_2, M_ext_2);

    // Segmentation
    cv::Mat im_segmented_1;
    cv::Mat im_segmented_2;

    cv::Mat output_segmentation_1, output_segmentation_2;
    std::vector<cv::Point2f> matched_points1;
    std::vector<cv::Point2f> matched_points2;
    // extract_features(im_gray_1, im_gray_2, &output_segmentation_1, &output_segmentation_2, &matched_points1, &matched_points2, 10000);

    cv::Point3f camera_pos_1 = get_camera_position(M_ext_1);
    cv::Point3f camera_pos_2 = get_camera_position(M_ext_2);

    cv::Mat M_transition_1 = compute_transition_matrix(M_int_1, M_ext_1);
    cv::Mat M_transition_2 = compute_transition_matrix(M_int_2, M_ext_2);
    cv::Mat M_transition_mire = compute_transition_matrix(M_int_mire, M_ext_mire);

    // std::vector<cv::Point3f> features_3D = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2);

    std::vector<cv::Point2f> points_mire_in_1;
    std::vector<cv::Point2f> points_mire_in_2;
    std::vector<cv::Point2f> points_mire_in_image_mire;

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
            cv::Mat p_mire = M_transition_mire * p;
            p_m1 /= p_m1.at<double>(2, 0);
            p_m2 /= p_m2.at<double>(2, 0);
            p_mire /= p_mire.at<double>(2, 0);

            points_mire_in_1.push_back(cv::Point2f(p_m1.at<double>(0, 0), p_m1.at<double>(1, 0)));
            points_mire_in_2.push_back(cv::Point2f(p_m2.at<double>(0, 0), p_m2.at<double>(1, 0)));
            points_mire_in_image_mire.push_back(cv::Point2f(p_mire.at<double>(0, 0), p_mire.at<double>(1, 0)));
        }
    }

    cv::Mat H_mire_to_1 = cv::findHomography(points_mire_in_image_mire, points_mire_in_1, cv::RANSAC);
    cv::Mat H_mire_to_2 = cv::findHomography(points_mire_in_image_mire, points_mire_in_2, cv::RANSAC);

    cv::Mat mire_image_in_1;
    cv::Mat mire_image_in_2;

    // cv::Mat mire_image_in_2 = cv::Mat::zeros(846, 1504, CV_32FC3);

    cv::Mat H_inv = H_mire_to_2.inv();

    warpPerspective(im_BGR_mire, mire_image_in_1, H_mire_to_1, im_BGR_2.size());
    warpPerspective(im_BGR_mire, mire_image_in_2, H_mire_to_2, im_BGR_2.size());

    imshow("Mire in image 1", mire_image_in_1);
    imshow("Mire in image 2", mire_image_in_2);

    cv::Mat im_segmentee_diff_1;
    im_segmentee_diff_1 = im_BGR_1_clean.clone();
    cv::Mat im_segmentee_diff_2;
    im_segmentee_diff_2 = im_BGR_2_clean.clone();

    float eps_diff_0 = 0.10f;
    float eps_diff_1 = 0.1f;
    float eps_diff_2 = 0.1f;

    im_segmentee_diff_1.convertTo(im_segmentee_diff_1, CV_32FC3, 255);
    im_segmentee_diff_2.convertTo(im_segmentee_diff_2, CV_32FC3, 255);
    im_BGR_1_clean.convertTo(im_BGR_1_clean, CV_32FC3, 255);
    im_BGR_2_clean.convertTo(im_BGR_2_clean, CV_32FC3, 255);

    for (int i = 0; i < im_BGR_mire.rows; ++i)
    {
        for (int j = 0; j < im_BGR_mire.cols; ++j)
        {
            if (((abs(im_BGR_1_clean.at<cv::Vec3f>(i, j)[0] - mire_image_in_1.at<cv::Vec3f>(i, j)[0]) < eps_diff_0) && (abs(im_BGR_1_clean.at<cv::Vec3f>(i, j)[1] - mire_image_in_1.at<cv::Vec3f>(i, j)[1]) < eps_diff_1) && (abs(im_BGR_1_clean.at<cv::Vec3f>(i, j)[2] - mire_image_in_1.at<cv::Vec3f>(i, j)[2]) < eps_diff_2)) || out_of_rectangle(i, j, M_transition_1))
            {
                im_segmentee_diff_1.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }

            if (((abs(im_BGR_2_clean.at<cv::Vec3f>(i, j)[0] - mire_image_in_2.at<cv::Vec3f>(i, j)[0]) < eps_diff_0) && (abs(im_BGR_2_clean.at<cv::Vec3f>(i, j)[1] - mire_image_in_2.at<cv::Vec3f>(i, j)[1]) < eps_diff_1) && (abs(im_BGR_2_clean.at<cv::Vec3f>(i, j)[2] - mire_image_in_2.at<cv::Vec3f>(i, j)[2]) < eps_diff_2)) || out_of_rectangle(i, j, M_transition_2))
            {
                im_segmentee_diff_2.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
        }
    }
    std::cout << im_BGR_2_clean.rows << "  " << im_BGR_2_clean.cols << std::endl;
    std::cout << im_segmentee_diff_1.rows << "  " << im_segmentee_diff_1.cols << std::endl;
    std::cout << im_BGR_mire.rows << "  " << im_BGR_mire.cols << std::endl;
    std::cout << "kg" << std::endl;

    cv::Mat im_segmentee_1;
    im_segmentee_1 = im_segmentee_diff_1; //.clone();
    cv::Mat im_segmentee_2;
    im_segmentee_2 = im_segmentee_diff_2; //.clone();

    for (unsigned int i = 10; i < im_BGR_mire.rows - 10; ++i)
    {
        for (unsigned int j = 10; j < im_BGR_mire.cols - 10; ++j)
        {
            int count_1 = 0;
            int count_2 = 0;
            for (int k = -10; k <= 10; k++)
            {
                for (int l = -10; l <= 10; l++)
                {
                    if (norm(im_segmentee_diff_1.at<cv::Vec3f>(i + k, j + l)) < 0.1f)
                        count_1 += 1;
                    if (norm(im_segmentee_diff_2.at<cv::Vec3f>(i + k, j + l)) < 0.1f)
                        count_2 += 1;
                }
            }
            if (count_1 > 21 * 21 * 3 / 8)
            {
                im_segmentee_diff_1.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
            if (count_2 > 21 * 21 * 3 / 8)
            {
                im_segmentee_diff_2.at<cv::Vec3f>(i, j) = cv::Vec3f(0.0f, 0.0f, 0.0f);
            }
        }
    }
    im_segmentee_diff_1.convertTo(im_segmentee_diff_1, CV_8UC3, 255);
    im_segmentee_diff_2.convertTo(im_segmentee_diff_2, CV_8UC3, 255);
    im_BGR_1_clean.convertTo(im_BGR_1_clean, CV_8UC3, 255);
    im_BGR_2_clean.convertTo(im_BGR_2_clean, CV_8UC3, 255);
    imshow("Image 1 segmentée", im_segmentee_diff_1);
    imshow("Image 2 segmentée", im_segmentee_diff_2);

    extract_features(im_segmentee_diff_1, im_segmentee_diff_2, &output_segmentation_1, &output_segmentation_2, &matched_points1, &matched_points2, 10000);

    std::vector<cv::Point2f> matched_transformed_1;
    std::vector<cv::Point2f> matched_transformed_2;
    std::cout << "Nombres matches 1 : " << matched_points1.size() << " | 2 : " << matched_points2.size() << std::endl;
    for (int i = 0; i < matched_points1.size(); i++)
    {
        circle(im_gray_2, matched_points1[i], 1, cv::Scalar(0, 255, 0), 2);
        circle(im_gray_2, matched_points2[i], 1, cv::Scalar(0, 0, 255), 2);
    }

    imshow("Features image 1", im_gray_2);
    imshow("Features image 2", im_gray_2);

    float eps = 100.0f;
    for (int i = 0; i < matched_points1.size(); i++)
    {
        cv::Mat p(3, 1, CV_64F);
        p.at<double>(0, 0) = matched_points1[i].x;
        p.at<double>(1, 0) = matched_points1[i].y;
        p.at<double>(2, 0) = 1.0f;

        cv::Mat p_m = H_mire_to_2 * p;

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