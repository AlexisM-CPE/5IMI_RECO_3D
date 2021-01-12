#include "points_detection.hpp"
#include "grid_creation.hpp"
#include "Point_Mire.hpp"
#include "features_extraction.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include "feature_location.hpp"

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // Declare the output variables
    Mat canny_edges_gray, im_hough_lines, im_hough_segments;

    // Loads an image
    Mat im_gray = imread("data/saliere/1.jpg", IMREAD_GRAYSCALE);
    Mat im_gray2 = imread("data/saliere/2.jpg", IMREAD_GRAYSCALE);
    Mat im_BGR = imread("data/saliere/1.jpg", IMREAD_COLOR);
    Mat im_BGR2 = imread("data/saliere/2.jpg", IMREAD_COLOR);

    if (!im_gray.data)
    {
        std::cout << "No image data" << std::endl;
        return -1;
    }

    // Convert CV_8UC3 to CV32FC3
    im_BGR.convertTo(im_BGR, CV_32FC3, 1.0 / 255.0);
    // Convert BGR image to HSV
    Mat im_HSV;
    cvtColor(im_BGR, im_HSV, COLOR_BGR2HSV);

    // Edge detection
    Canny(im_gray, canny_edges_gray, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(canny_edges_gray, im_hough_lines, COLOR_GRAY2BGR);
    im_hough_segments = im_hough_lines.clone();

    // Standard Hough Line Transform (lines)
    vector<Vec2f> lines;                                            // will hold the results of the detection
    HoughLines(canny_edges_gray, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

    // Draw the lines
    for (unsigned int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point2f pt1, pt2;
        float a = cos(theta), b = sin(theta);
        float x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 10000 * (-b));
        pt1.y = cvRound(y0 + 10000 * (a));
        pt2.x = cvRound(x0 - 10000 * (-b));
        pt2.y = cvRound(y0 - 10000 * (a));
        line(im_hough_lines, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Probabilistic Line Transform (segments)
    vector<Vec4i> segments;                                              // will hold the results of the detection
    HoughLinesP(canny_edges_gray, segments, 1, CV_PI / 180, 50, 30, 10); // runs the actual detection

    vector<float> angles = get_angles(segments);
    vector<int> labels = kmeans(angles);

    vector<Vec4i> segmentsV;
    vector<Vec4i> segmentsH;

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
        Vec4i points_segment = segmentsH[i];
        line(im_hough_segments, Point(points_segment[0], points_segment[1]), Point(points_segment[2], points_segment[3]), Scalar(0, 255, 0), 3, LINE_AA);
    }
    for (unsigned int i = 0; i < segmentsV.size(); i++)
    {
        Vec4i points_segment = segmentsV[i];
        line(im_hough_segments, Point(points_segment[0], points_segment[1]), Point(points_segment[2], points_segment[3]), Scalar(0, 0, 255), 3, LINE_AA);
    }

    // Computing the intersections between the segments
    vector<Point2f> intersection_points_segments;
    for (auto lineV : segmentsV)
    {
        for (auto lineH : segmentsH)
        {
            Point2f o1(lineH[0], lineH[1]);
            Point2f p1(lineH[2], lineH[3]);
            Point2f o2(lineV[0], lineV[1]);
            Point2f p2(lineV[2], lineV[3]);
            Point2f r;

            if (intersection(o1, p1, o2, p2, r))
            {
                intersection_points_segments.push_back(r);
            }
        }
    }

    // Merging the close intersection points
    float r = 10.0f;
    float eps_dist_lines = 5.0f;

    vector<Point2f> merged_points = merge_close_points(intersection_points_segments, r);
    vector<Point2f> intersection_points;
    // Drawing the points and keeping only the ones close to a line
    for (unsigned int i = 0; i < merged_points.size(); ++i)
    {
        int x = (merged_points[i]).x;
        int y = (merged_points[i]).y;
        vector<float> distances;

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
            circle(im_hough_segments, Point(x, y), 1, Scalar(255, 0, 0), 2);
            circle(im_hough_lines, Point(x, y), 1, Scalar(255, 0, 0), 2);
            circle(im_hough_segments, Point(x, y), 15 / 2, Scalar(255, 0, 0), 1);

            circle(im_hough_lines, Point(x, y), 15 / 2, Scalar(255, 0, 0), 1);
            int c = find_color(im_HSV, Point(x, y));
            switch (c)
            {
            case NO_COLOR:
                circle(im_BGR, Point(x, y), 1, Scalar(0, 0, 0), 2);
                break;
            case MAGENTA:
                circle(im_BGR, Point(x, y), 1, Scalar(255, 0, 255), 2);
                break;
            case YELLOW:
                circle(im_BGR, Point(x, y), 1, Scalar(0, 255, 255), 2);
                break;
            case CYAN:
                circle(im_BGR, Point(x, y), 1, Scalar(255, 255, 0), 2);
                break;
            case WHITE:
                circle(im_BGR, Point(x, y), 1, Scalar(255, 255, 255), 2);
                break;
            }
            intersection_points.push_back(Point2f(x, y));
        }

        // Plotting the other points
        else
        {
            circle(im_hough_segments, Point(x, y), 1, Scalar(255, 255, 0), 2);
            circle(im_hough_segments, Point(x, y), 15 / 2, Scalar(255, 255, 0), 1);
        }
    }
    imshow("Source", im_BGR);
    // Show results
    //imshow("Source", im_BGR);
    //resizeWindow("Source", im_BGR.cols, im_BGR.rows);
    //imshow("Detected Segments (in green) and points", im_hough_segments);
    //imshow("Detected Lines (in red), usefull points (in blue) and discarded points (in green)", im_hough_lines);
    std::vector<Point_Mire *> points_grille = find_pos(im_HSV, intersection_points);

    vector<vector<Point3f>> object_points = extract_object_points(points_grille);
    vector<vector<Point2f>> image_points = extract_image_points(points_grille);
    std::cout << "size : " << points_grille.size() << std::endl;
    Mat cameraMatrix(3, 3, CV_32FC1);

    cameraMatrix.at<float>(0, 2) = im_BGR.rows / 2;
    cameraMatrix.at<float>(1, 2) = im_BGR.cols / 2;

    float f = 4;
    float s = 0.0014;

    cameraMatrix.at<float>(2, 2) = 1;
    cameraMatrix.at<float>(1, 1) = f / s;
    cameraMatrix.at<float>(0, 0) = f / s;

    //Mat cameraMatrix;
    Mat distCoeffs;

    vector<Mat> rvecs;
    vector<Mat> tvecs;

    Mat rot(3, 3, CV_64F);
    Mat rot_tr(3, 3, CV_64F);

    calibrateCamera(object_points, image_points, im_BGR.size(), cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO);

    Rodrigues(rvecs[0], rot);
    transpose(rot, rot_tr);

    Mat pos_camera(3,1,CV_64F);

    pos_camera = -rot_tr * tvecs[0];

    Mat t_obj(3, 3, CV_64F);
    t_obj = -rot * pos_camera;

    std::cout << "____" << std::endl;
    std::cout << t_obj.at<double>(0, 0) << std::endl;
    std::cout << t_obj.at<double>(1, 0) << std::endl;
    std::cout << t_obj.at<double>(2, 0) << std::endl;
    std::cout << "____" << std::endl;

    std::cout << tvecs[0].at<double>(0, 0) << std::endl;
    std::cout << tvecs[0].at<double>(1, 0) << std::endl;
    std::cout << tvecs[0].at<double>(2, 0) << std::endl;
    std::cout << "____" << std::endl;

    float phi = -atan(rot_tr.at<double>(1, 2) / rot_tr.at<double>(2, 2));
    float gamma = -atan(rot_tr.at<double>(0, 1) / rot_tr.at<double>(0, 0));
    float omega = atan(rot_tr.at<double>(0, 2) / (-rot_tr.at<double>(1, 2) * sin(phi) + rot_tr.at<double>(2, 2) * cos(phi)));

    std::cout << "Phi : " << phi << std::endl;
    std::cout << "Gamma : " << gamma << std::endl;
    std::cout << "Omega : " << omega << std::endl;
    std::cout << "____" << std::endl;

    std::cout << "X : " << pos_camera.at<double>(0, 0) << std::endl;
    std::cout << "Y : " << pos_camera.at<double>(1, 0) << std::endl;
    std::cout << "Z : " << pos_camera.at<double>(2, 0) << std::endl;



    Mat M_int_2 = create_M_int(cameraMatrix);
    Mat M_ext_2 = create_M_ext(rvecs, tvecs);
    Mat M_trans = compute_transition_matrix(M_int_2, M_ext_2);
    Point3f cam_pos_2 = get_camera_position(rvecs, tvecs);

    Point2f pt_8_1_image(512, 251);
    Point2f pt_15_8_image(613, 589);

    Point3f pt_15_8_world = image_to_grid_plan(pt_15_8_image, M_trans);

    std::cout << "camera X v2 : " << cam_pos_2.x << std::endl;
    std::cout << "camera Y v2 : " << cam_pos_2.y << std::endl;
    std::cout << "camera Z v2 : " << cam_pos_2.z << std::endl;

    std::cout << "Point X v2 : " << pt_15_8_world.x/12.4f << std::endl;
    std::cout << "Point Y v2 : " << pt_15_8_world.y/12.4f << std::endl;
    std::cout << "Point Z v2 : " << pt_15_8_world.z << std::endl;





    Mat image_points_output;
    Mat jacobian;
    double aspectRatio = 16.0f / 9;

    Point3f p_c = Point3f(pos_camera.at<double>(0, 0), pos_camera.at<double>(1, 0), pos_camera.at<double>(2, 0));
    Point3f d = Point3f(8 * 12.4, 8 * 12.4, 0.0f) - p_c;
    Point3f p_c2 = p_c + d / 10.0f;

    object_points[0].push_back(p_c2);

    projectPoints(object_points.front(), rvecs.front(), tvecs.front(), cameraMatrix, distCoeffs, image_points_output, jacobian, aspectRatio);

    for (int i = 0; i < image_points_output.rows - 1; i++)
    {
        auto p = image_points_output.at<Point2f>(i);
        circle(im_BGR, Point(p.x, p.y), 1, Scalar(0, 255, 0), 2);
        imshow("Source", im_BGR);
    }

    Mat imageo1, imageo2;
    extract_features(im_gray, im_gray2, &imageo1, &imageo2, 1000);

    // //FAST(image, &keypointsD, threshold, true);
    // drawKeypoints(im_gray, feat, imageKey);

    // Ptr<StereoSGBM> BMState = cv::StereoSGBM::create(0, 8 * 16, 3, 200, 400, 0, 15, 7, 200, 2, StereoSGBM::MODE_HH);
    // BMState->compute(im_gray, im_gray2, imageKey);
    //imageKey = (imageKey - min) * 255 / (max - min);
    //cv::normalize(imageKey, imageKey, 0, 256, CV_MMX);
    //normalize(imageKey, imageKey, 0, 255, NORM_MINMAX, CV_8U);
    imshow("keypoints", imageo1);

    imshow("keypoints2", imageo2);

    // // Flann needs the descriptors to be of type CV_32F
    // descriptors_1.convertTo(descriptors_1, CV_8UC1);
    // descriptors_2.convertTo(descriptors_2, CV_8UC1);

    // std::cout << descriptors_1.rows << ", " << descriptors_1.cols << std::endl;
    // std::cout << descriptors_2.rows << ", " << descriptors_2.cols << std::endl;
    // std::cout << "_________________________________" <<std::endl;

    // Mat im_out;
    // drawKeypoints(im_gray, key_points_1, im_out, Scalar(0,0,255));

    /*
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    matcher.match( descriptors_1, descriptors_2, matches );

    double max_dist = 0; double min_dist = 100;


    //-- Quick calculation of max and min distances between keypoints
    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
    vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1.rows; i++ )
    {
        if( matches[i].distance < 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }
    
    vector< Point2f > obj;
    vector< Point2f > scene;


    for( int i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( key_points_1[ good_matches[i].queryIdx ].pt );
        scene.push_back( key_points_1[ good_matches[i].trainIdx ].pt );
    }


    // Find the Homography Matrix
    Mat H = findHomography( obj, scene, RANSAC );
    // Use the Homography Matrix to warp the images
    cv::Mat result;
    warpPerspective(im_gray,result,H,Size(im_gray.cols+im_gray2.cols,im_gray.rows));
    cv::Mat half(result,cv::Rect(0,0,im_gray2.cols,im_gray2.rows));
    im_gray2.copyTo(half);
    imshow( "Result", result );
*/

    // // Feature matching
	// BFMatcher BF = BFMatcher(NORM_HAMMING);
	// std::vector<vector<DMatch> > matches;
	// BF.knnMatch(descriptors_1, descriptors_2, matches, 2);

	// std::vector<DMatch> match1;
	// std::vector<DMatch> match2;

	// for (int i = 0; i < matches.size(); i++)
	// {
	// 	match1.push_back(matches[i][0]);
	// 	match2.push_back(matches[i][1]);
	// }

    // Mat img_matches1, img_matches2;
	// drawMatches(im_gray, key_points_1, im_gray2, key_points_2, match1, img_matches1);
	// drawMatches(im_gray, key_points_1, im_gray2, key_points_2, match2, img_matches2);

	// imshow("test2", img_matches1);
	// imshow("test4", img_matches2);


    // imshow("out", im_out);
    // Wait and Exit
    waitKey(0);
    return 0;
}