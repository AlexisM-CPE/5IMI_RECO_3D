#include "utils.hpp"

#include <fstream>
#include <cmath>

#define offset 0

int find_points_mire(cv::Mat &im_gray, cv::Mat &im_BGR, std::vector<std::vector<cv::Point3f>> &object_points, std::vector<std::vector<cv::Point2f>> &image_points, std::string name)
{
    cv::Mat canny_edges_gray, im_hough_lines, im_hough_segments;
    if (!im_gray.data)
    {
        std::cout << "No image data" << std::endl;
        return -1;
    }

    // Convert CV_8UC3 to CV32FC3
    im_BGR.convertTo(im_BGR, CV_32FC3, 1.0 / 255.0);
    // Convert BGR image to HSV
    cv::Mat im_HSV;
    cvtColor(im_BGR, im_HSV, cv::COLOR_BGR2HSV);

    // Edge detection
    Canny(im_gray, canny_edges_gray, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(canny_edges_gray, im_hough_lines, cv::COLOR_GRAY2BGR);
    im_hough_segments = im_hough_lines.clone();

    // Standard Hough Line Transform (lines)
    std::vector<cv::Vec2f> lines;                                   // will hold the results of the detection
    HoughLines(canny_edges_gray, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

    // Draw the lines
    for (unsigned int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        cv::Point2f pt1, pt2;
        float a = cos(theta), b = sin(theta);
        float x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 10000 * (-b));
        pt1.y = cvRound(y0 + 10000 * (a));
        pt2.x = cvRound(x0 - 10000 * (-b));
        pt2.y = cvRound(y0 - 10000 * (a));
        line(im_hough_lines, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    // Probabilistic Line Transform (segments)
    std::vector<cv::Vec4i> segments;                                     // will hold the results of the detection
    HoughLinesP(canny_edges_gray, segments, 1, CV_PI / 180, 30, 30, 10); // runs the actual detection

    std::vector<float> angles = get_angles(segments);
    std::vector<int> labels = kmeans(angles);

    std::vector<cv::Vec4i> segmentsV;
    std::vector<cv::Vec4i> segmentsH;

    for (unsigned int k = 0; k < segments.size(); k++)
    {
        if (labels[k] == 1)
        {
            segmentsV.push_back(segments[k]);
        }
        else if (labels[k] == 0)
        {
            segmentsH.push_back(segments[k]);
        }
    }

    // Draw the lines
    for (unsigned int i = 0; i < segmentsH.size(); i++)
    {
        cv::Vec4i points_segment = segmentsH[i];
        line(im_hough_segments, cv::Point(points_segment[0], points_segment[1]), cv::Point(points_segment[2], points_segment[3]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }
    for (unsigned int i = 0; i < segmentsV.size(); i++)
    {
        cv::Vec4i points_segment = segmentsV[i];
        line(im_hough_segments, cv::Point(points_segment[0], points_segment[1]), cv::Point(points_segment[2], points_segment[3]), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }

    // Computing the intersections between the segments
    std::vector<cv::Point2f> intersection_points_segments;
    for (auto lineV : segmentsV)
    {
        for (auto lineH : segmentsH)
        {
            cv::Point2f o1(lineH[0], lineH[1]);
            cv::Point2f p1(lineH[2], lineH[3]);
            cv::Point2f o2(lineV[0], lineV[1]);
            cv::Point2f p2(lineV[2], lineV[3]);
            cv::Point2f r;

            if (intersection(o1, p1, o2, p2, r))
            {
                intersection_points_segments.push_back(r);
            }
        }
    }

    // Merging the close intersection points
    float r = 10.0f;
    float eps_dist_lines = 5.0f;

    std::vector<cv::Point2f> merged_points = merge_close_points(intersection_points_segments, r);
    std::vector<cv::Point2f> intersection_points;
    // Drawing the points and keeping only the ones close to a line
    for (unsigned int i = 0; i < merged_points.size(); ++i)
    {
        int x = (merged_points[i]).x;
        int y = (merged_points[i]).y;
        std::vector<float> distances;

        // Computing the distance to every line
        for (unsigned int k = 0; k < lines.size(); k++)
        {
            float rho = lines[k][0], theta = lines[k][1];
            float x1, x2, y1, y2;
            double a = cos(theta), b = sin(theta);
            double x0 = a * rho, y0 = b * rho;
            x1 = cvRound(x0 + 10000 * (-b));
            y1 = cvRound(y0 + 10000 * (a));
            x2 = cvRound(x0 - 10000 * (-b));
            y2 = cvRound(y0 - 10000 * (a));

            float dist = abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
            distances.push_back(dist);
        }

        // Keeping only the points close to a line
        if (*min_element(distances.begin(), distances.end()) <= eps_dist_lines)
        {
            if (name != "None")
            {
                circle(im_hough_segments, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), 2);
                circle(im_hough_lines, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), 2);
                circle(im_hough_segments, cv::Point(x, y), 15 / 2, cv::Scalar(255, 0, 0), 1);

                circle(im_hough_lines, cv::Point(x, y), 15 / 2, cv::Scalar(255, 0, 0), 1);
                int c = find_color(im_HSV, cv::Point(x, y));
                switch (c)
                {
                case NO_COLOR:
                    circle(im_BGR, cv::Point(x, y), 1, cv::Scalar(0, 0, 0), 2);
                    break;
                case MAGENTA:
                    circle(im_BGR, cv::Point(x, y), 1, cv::Scalar(255, 0, 255), 2);
                    break;
                case YELLOW:
                    circle(im_BGR, cv::Point(x, y), 1, cv::Scalar(0, 255, 255), 2);
                    break;
                case CYAN:
                    circle(im_BGR, cv::Point(x, y), 1, cv::Scalar(255, 255, 0), 2);
                    break;
                case WHITE:
                    circle(im_BGR, cv::Point(x, y), 1, cv::Scalar(255, 255, 255), 2);
                    break;
                }
            }
            intersection_points.push_back(cv::Point2f(x, y));
        }

        // Plotting the other points
        else
        {
            circle(im_hough_segments, cv::Point(x, y), 1, cv::Scalar(255, 255, 0), 2);
            circle(im_hough_segments, cv::Point(x, y), 15 / 2, cv::Scalar(255, 255, 0), 1);
        }
    }

    if (name != "None")
    {
        cv::imshow(name, im_BGR);
        cv::imshow(name + " Hough", im_hough_segments);
    }
    std::vector<Point_Mire *> points_grille = find_pos(im_HSV, intersection_points);

    object_points = extract_object_points(points_grille);
    image_points = extract_image_points(points_grille);

    return 0;
}

void Calibrate(cv::Mat &im_gray, cv::Mat &im_BGR, std::vector<std::vector<cv::Point3f>> &object_points, std::vector<std::vector<cv::Point2f>> &image_points, cv::Mat &cameraMatrix, cv::Mat distCoeffs, cv::Mat &M_int, cv::Mat &M_ext, std::string name)
{

    float f = 4;
    float s = 0.0014;

    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);

    cameraMatrix.at<float>(0, 2) = im_BGR.rows / 2;
    cameraMatrix.at<float>(1, 2) = im_BGR.cols / 2;
    cameraMatrix.at<float>(1, 1) = f / s;
    cameraMatrix.at<float>(0, 0) = f / s;

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;

    calibrateCamera(object_points, image_points, im_BGR.size(), cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

    cv::Mat rot(3, 3, CV_64F);
    cv::Mat rot_tr(3, 3, CV_64F);

    Rodrigues(rvecs[0], rot);

    M_ext = create_M_ext(rvecs, tvecs);
    M_int = create_M_int(cameraMatrix);

    cv::Mat output_image;

    // cv::undistort(im_BGR, output_image, cameraMatrix, distCoeffs);

    //cv::imshow("Undistort" + name, output_image);

    // im_BGR = output_image;

    if (name != "None")
    {
        cv::Point3f cam_pos = get_camera_position(M_ext);

        cv::Mat image_points_output;
        cv::Mat jacobian;
        double aspectRatio = 1.0f;

        cv::Point3f p_c = cv::Point3f(cam_pos.x, cam_pos.y, cam_pos.z);
        cv::Point3f d = cv::Point3f(8 * 12.375, 8 * 12.375, 0.0f) - p_c;
        cv::Point3f p_c2 = p_c + d / 10.0f;

        object_points[0].push_back(p_c2);

        projectPoints(object_points.front(), rvecs.front(), tvecs.front(), cameraMatrix, distCoeffs, image_points_output, jacobian, aspectRatio);

        for (int i = 0; i < image_points_output.rows; i++)
        {
            auto p = image_points_output.at<cv::Point2f>(i);
            cv::circle(im_BGR, cv::Point(p.x, p.y), 1, cv::Scalar(0, 255, 0), 2);
        }
        for (int i = 0; i < image_points[0].size(); i++)
        {
            auto p = image_points[0][i];
            cv::circle(im_BGR, cv::Point(p.x, p.y), 1, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow(name, im_BGR);

        cv::Point3f c1(0.0f, 0.0f, 0.0f);
        cv::Point3f c2(16.0f * 12.375f, 0.0f, 0.0f);
        cv::Point3f c3(0.0f, 16.0f * 12.375f, 0.0f);
        cv::Point3f c4(16.0f * 12.375f, 16.0f * 12.375f, 0.0f);

        std::vector<cv::Point3f> coins;
        coins.push_back(c1);
        coins.push_back(c2);
        coins.push_back(c3);
        coins.push_back(c4);

        std::vector<std::vector<cv::Point3f>> coins_vect;
        coins_vect.push_back(coins);

        cv::Mat image_coins_output;

        projectPoints(coins_vect.front(), rvecs.front(), tvecs.front(), cameraMatrix, distCoeffs, image_coins_output, jacobian, aspectRatio);

        cv::Mat im_coins = im_BGR.clone();

        for (int i = 0; i < image_coins_output.rows; i++)
        {
            auto p = image_coins_output.at<cv::Point2f>(i);
            cv::circle(im_BGR, cv::Point(p.x, p.y), 1, cv::Scalar(255, 0, 0), 2);
        }

        imshow("coins " + name, im_BGR);
    }
}

void create_cloud_file(std::vector<cv::Point3f> points, std::string filename)
{
    std::ofstream cloud_file(filename);
    for (auto point : points)
    {
        cloud_file << point.x << " " << point.y << " " << point.z << std::endl;
    }
}

std::vector<cv::Point3f> read_cloud_file(std::string filename)
{
    std::vector<cv::Point3f> points;
    std::ifstream cloud_file(filename);
    std::string line;

    while (std::getline(cloud_file, line))
    {
        std::istringstream iss(line);
        float x, y, z;
        if (!(iss >> x >> y >> z))
        {
            break;
        }

        cv::Point3f p(x, y, z);
        points.push_back(p);
    }
    return points;
}

bool out_of_rectangle(int i, int j, cv::Mat M_transition)
{
    cv::Mat c0(4, 1, CV_64F);
    cv::Mat c1(4, 1, CV_64F);
    cv::Mat c2(4, 1, CV_64F);
    cv::Mat c3(4, 1, CV_64F);

    c0.at<double>(0, 0) = 0.0f + offset * 12.4f;
    c0.at<double>(1, 0) = 0.0f + offset * 12.4f;
    c0.at<double>(2, 0) = 0.0f;
    c0.at<double>(3, 0) = 1.0f;

    c1.at<double>(0, 0) = (16.0f + offset) * 12.4f;
    c1.at<double>(1, 0) = 0.0f + offset * 12.4f;
    c1.at<double>(2, 0) = 0.0f;
    c1.at<double>(3, 0) = 1.0f;

    c2.at<double>(0, 0) = 0.0f + offset * 12.4f;
    c2.at<double>(1, 0) = (16.0f + offset) * 12.4f;
    c2.at<double>(2, 0) = 0.0f;
    c2.at<double>(3, 0) = 1.0f;

    c3.at<double>(0, 0) = (16.0f + offset) * 12.4f;
    c3.at<double>(1, 0) = (16.0f + offset) * 12.4f;
    c3.at<double>(2, 0) = 0.0f;
    c3.at<double>(3, 0) = 1.0f;

    cv::Mat p0 = M_transition * c0;
    cv::Mat p1 = M_transition * c1;
    cv::Mat p2 = M_transition * c2;
    cv::Mat p3 = M_transition * c3;

    cv::Point2f t0(p0.at<double>(0, 0) / p0.at<double>(2, 0), p0.at<double>(1, 0) / p0.at<double>(2, 0));
    cv::Point2f t1(p1.at<double>(0, 0) / p1.at<double>(2, 0), p1.at<double>(1, 0) / p1.at<double>(2, 0));
    cv::Point2f t2(p2.at<double>(0, 0) / p2.at<double>(2, 0), p2.at<double>(1, 0) / p2.at<double>(2, 0));
    cv::Point2f t3(p3.at<double>(0, 0) / p3.at<double>(2, 0), p3.at<double>(1, 0) / p3.at<double>(2, 0));

    cv::Point2f p(i, j);

    float a1 = acos(((p - t0) / norm(p - t0)).dot((p - t1) / norm(p - t1)));
    float a2 = acos(((p - t1) / norm(p - t1)).dot((p - t2) / norm(p - t2)));
    float a3 = acos(((p - t2) / norm(p - t2)).dot((p - t3) / norm(p - t3)));
    float a4 = acos(((p - t3) / norm(p - t3)).dot((p - t0) / norm(p - t0)));

    float eps = M_PI / 24;

    if (2 * M_PI - (a1 + a2 + a3 + a4) > eps)
        return true;
    else
        return false;
}