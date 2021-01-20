#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include "segmentation.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include "ProgressBar.hpp"

#include "feature_location.hpp"
#include <filesystem>

#include "utils.hpp"
#include <iostream>
#include <cmath>
#include <omp.h>

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
void compute_cloud_image(std::string filename_im_1, std::string filename_im_2, std::string filename_cloud)
{
    // Loads an image

    //cv::Mat im_gray_1 = imread("data/origami/1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_mire = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_mire = imread("data/mario/2.jpg", cv::IMREAD_COLOR);

    cv::Mat im_gray_1 = imread(filename_im_1, cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_1 = imread(filename_im_1, cv::IMREAD_COLOR);

    //cv::Mat im_gray_2 = imread("data/mario/2.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat im_gray_2 = imread(filename_im_2, cv::IMREAD_GRAYSCALE);
    cv::Mat im_BGR_2 = imread(filename_im_2, cv::IMREAD_COLOR);

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

    // imshow("Mire in image 1", mire_image_in_1);
    // imshow("Mire in image 2", mire_image_in_2);

    cv::Mat im_segmentee_diff_1;
    im_segmentee_diff_1 = im_BGR_1_clean.clone();
    cv::Mat im_segmentee_diff_2;
    im_segmentee_diff_2 = im_BGR_2_clean.clone();

    float eps_diff_0 = 0.10f;
    float eps_diff_1 = 0.1f;
    float eps_diff_2 = 0.1f;

    im_segmentee_diff_1.convertTo(im_segmentee_diff_1, CV_32FC3, 1.0f / 255.0f);
    im_segmentee_diff_2.convertTo(im_segmentee_diff_2, CV_32FC3, 1.0f / 255.0f);
    im_BGR_1_clean.convertTo(im_BGR_1_clean, CV_32FC3, 1.0f / 255.0f);
    im_BGR_2_clean.convertTo(im_BGR_2_clean, CV_32FC3, 1.0f / 255.0f);

#pragma omp parallel for schedule(dynamic, 1)
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

    cv::Mat im_segmentee_1 = cv::Mat::zeros(846, 1504, im_segmentee_diff_1.type());
    cv::Mat im_segmentee_2 = cv::Mat::zeros(846, 1504, im_segmentee_diff_2.type());
    // cv::Mat im_segmentee_2 = im_segmentee_diff_2.clone();

    int x_min_1, x_max_1, y_min_1, y_max_1, x_min_2, x_max_2, y_min_2, y_max_2;
    get_box(x_min_1, x_max_1, y_min_1, y_max_1, M_transition_1);
    get_box(x_min_2, x_max_2, y_min_2, y_max_2, M_transition_2);
    // std::cout << x_min_1 << "  " << x_max_1 << "  " << y_min_1 << "  " << y_max_1 << std::endl;
    // std::cout << x_min_2 << "  " << x_max_2 << "  " << y_min_2 << "  " << y_max_2 << std::endl;
    // imshow("Image 1 diff segmentee", im_segmentee_diff_1);
    // imshow("Image 2 diff segmentee", im_segmentee_diff_2);

#pragma omp parallel for schedule(dynamic, 1)
    for (int i = 10; i < 836; ++i)
    {
        for (int j = 10; j < 1494; ++j)
        {
            int count_1 = 0;
            int count_2 = 0;

            for (int k = -10; k <= 10; k++)
            {
                for (int l = -10; l <= 10; l++)
                {
                    if (norm(im_segmentee_diff_1.at<cv::Vec3f>(i + k, j + l)) < 0.1f)
                        count_1++;
                    if (norm(im_segmentee_diff_2.at<cv::Vec3f>(i + k, j + l)) < 0.1f)
                        count_2 += 1;
                }
            }
            // std::cout << count_1 << std::endl;
            if (count_1 <= 21 * 21 * 3 / 8)
                im_segmentee_1.at<cv::Vec3f>(i, j) = im_segmentee_diff_1.at<cv::Vec3f>(i, j);
            if (count_2 < 21 * 21 * 3 / 8)
                im_segmentee_2.at<cv::Vec3f>(i, j) = im_segmentee_diff_2.at<cv::Vec3f>(i, j);
        }
    }

    im_segmentee_1.convertTo(im_segmentee_1, CV_8UC3, 255);
    im_segmentee_2.convertTo(im_segmentee_2, CV_8UC3, 255);
    im_BGR_1_clean.convertTo(im_BGR_1_clean, CV_8UC3, 255);
    im_BGR_2_clean.convertTo(im_BGR_2_clean, CV_8UC3, 255);
    // imshow("Image 1 segmentee", im_segmentee_1);
    // imshow("Image 2 segmentee", im_segmentee_2);

    extract_features(im_segmentee_1, im_segmentee_2, &output_segmentation_1, &output_segmentation_2, &matched_points1, &matched_points2, 10000);
    std::vector<cv::Vec3i> colors;
    std::vector<int> features_index;

    // std::cout << camera_pos_1.x << "    " << camera_pos_1.y << "    " << camera_pos_1.z << std::endl;
    // std::cout << camera_pos_2.x << "    " << camera_pos_2.y << "    " << camera_pos_2.z << std::endl;
    std::vector<cv::Point3f> points_nuage = find_feature_3d_im1_im2(matched_points1, matched_points2, camera_pos_1, camera_pos_2, M_transition_1, M_transition_2, features_index);
    for (int i : features_index)
    {
        if (i % 7 == 0)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(0, 0, 255), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(0, 0, 255), 2);
        }

        else if (i % 7 == 1)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(0, 255, 255), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(0, 255, 255), 2);
        }
        else if (i % 7 == 2)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(255, 255, 255), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(255, 255, 255), 2);
        }
        else if (i % 7 == 3)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(0, 255, 0), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(0, 255, 0), 2);
        }
        else if (i % 7 == 4)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(255, 0, 255), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(255, 0, 255), 2);
        }
        else if (i % 7 == 5)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(255, 0, 0), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(255, 0, 0), 2);
        }
        else if (i % 7 == 6)
        {
            circle(im_segmentee_1, matched_points1[i], 1, cv::Scalar(255, 255, 0), 2);
            circle(im_segmentee_2, matched_points2[i], 1, cv::Scalar(255, 255, 0), 2);
        }

        cv::Vec3i c_bgr = (im_BGR_1_clean.at<cv::Vec3b>(int(matched_points1[i].y), int(matched_points1[i].x))); // + im_BGR_2_clean.at<cv::Vec3b>(int(matched_points2[i].x), int(matched_points2[i].y)));
        colors.push_back(cv::Vec3i(c_bgr[2], c_bgr[1], c_bgr[0]));
        // std::cout << c_bgr[2] << "    " << colors[i][0] << std::endl;
    }
    create_cloud_file_ply(points_nuage, colors, filename_cloud);

    // imshow("Features segmentees 1", im_segmentee_1);
    // imshow("Features segmentees 2", im_segmentee_2);

    std::vector<cv::Point2f> matched_transformed_1;
    std::vector<cv::Point2f> matched_transformed_2;
    // std::cout << "Nombres matches 1 : " << matched_points1.size() << " | 2 : " << matched_points2.size() << std::endl;
}
int main(int argc, char **argv)
{
    if (std::filesystem::exists("nuage_all_fleur.ply"))
        remove("nuage_all_fleur.ply");
    ProgressBar bar(std::cout, 21);
    bar.init();

    /*for (int i = 1; i < 14; i++)
    {
        //std::cout << "image : " << 3 * i << " , " << 3 * i + 1 << " , " << 3 * i + 2 << std::endl;
        compute_cloud_image("data/mario/" + std::to_string(3 * i) + ".jpg", "data/mario/" + std::to_string(3 * i + 1) + ".jpg", "nuage_all.ply");
        bar.update(i - 2, 40);
        compute_cloud_image("data/mario/" + std::to_string(3 * i + 1) + ".jpg", "data/mario/" + std::to_string(3 * i + 2) + ".jpg", "nuage_all.ply");
        bar.update(i - 2, 40);
        compute_cloud_image("data/mario/" + std::to_string(3 * i + 2) + ".jpg", "data/mario/" + std::to_string(3 * i) + ".jpg", "nuage_all.ply");
        bar.update(i - 2, 40);
    }*/

    for (int i = 1; i < 5; i++)
    {
        //std::cout << "image : " << 3 * i << " , " << 3 * i + 1 << " , " << 3 * i + 2 << std::endl;
        compute_cloud_image("data/fleur/" + std::to_string(i) + ".jpg", "data/fleur/" + std::to_string(i + 1) + ".jpg", "nuage_all_fleur.ply");
        bar.update(i - 2, 21);
        std::cout << i << std::endl;
    }

    std::cout << "]" << std::endl;
    // while (true)
    // {
    //     // Close and quit only when Escape is pressed
    //     int key = cv::waitKey(0);
    //     if (key == 27 || key == -1)
    //         break;
    // }
    return 0;
}