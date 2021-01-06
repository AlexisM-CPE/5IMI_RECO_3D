#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "points_detection.hpp"
#include "grid_creation.hpp"

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;



int main(int argc, char** argv)
{
    // Declare the output variables
    Mat dst, cdst, cdstP;
    const char* default_file = "data/saliere/test_1.jpg";
    const char* filename = argc >=2 ? argv[1] : default_file;
    // Loads an image
    Mat Gray_image = imread( samples::findFile( filename ), IMREAD_GRAYSCALE );
    // Check if image is loaded fine
    if(Gray_image.empty()){
        printf(" Error opening image\n");
        printf(" Program Arguments: [image_name -- default %s] \n", default_file);
        return -1;
    }


    // Edge detection
    Canny(Gray_image, dst, 50, 200, 3);
    // Copy edges to the images that will display the results in BGR
    cvtColor(dst, cdst, COLOR_GRAY2BGR);
    cdstP = cdst.clone();

    // Standard Hough Line Transform
    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(dst, lines, 1, CV_PI/180, 150, 0, 0 ); // runs the actual detection
    // Draw the lines
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 10000*(-b));
        pt1.y = cvRound(y0 + 10000*(a));
        pt2.x = cvRound(x0 - 10000*(-b));
        pt2.y = cvRound(y0 - 10000*(a));
        line( cdst, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
        circle(cdstP, pt1, 3, Scalar(0,255,0), 3);
        circle(cdstP, pt2, 3, Scalar(0,255,0), 3);
    }
    // Probabilistic Line Transform
    vector<Vec4i> linesP; // will hold the results of the detection
    HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 30, 10 ); // runs the actual detection
    // Draw the lines
    //vector<float> angles = get_angles(linesP);
    //vector<int> labels = kmeans(angles);
/*
    vector<Vec4i> linesV;
    vector<Vec4i> linesH;

    for (int k = 0 ; k < linesP.size(); k++){
        if (labels[k] == 1  )
            linesV.push_back(linesP[k]);
        else
            linesH.push_back(linesP[k]);

    }

*/
    for( size_t i = 0; i < linesP.size(); i++ )
    {
        Vec4i l = linesP[i];
        line( cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,255,0), 3, LINE_AA);
    }
    

    vector<Point2f> intersectionPoints;
    for (auto lineV : linesP){
        for (auto lineH : linesP){
            if (lineV != lineH){
                Point2f o1(lineH[0], lineH[1]);
                Point2f p1(lineH[2], lineH[3]);
                Point2f o2(lineV[0], lineV[1]);
                Point2f p2(lineV[2], lineV[3]);
                Point2f r = Point2f(0.0f, 0.0f);

                if (intersection(o1, p1, o2, p2, r)){
                    intersectionPoints.push_back(r);
                }
            }
            
        }
    }

	vector<Point2f> merged_points = merge_close_points(intersectionPoints, 10);
    float eps_dist_lines = 5.0f;
	for (int i = 0; i < merged_points.size(); ++i) {
		int x = (merged_points[i]).x;
		int y = (merged_points[i]).y;
        vector<float> distances;
        for (int k = 0 ; k < lines.size() ; k++){
            float rho = lines[k][0], theta = lines[k][1];
            float x1, x2, y1, y2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            x1 = cvRound(x0 + 10000*(-b));
            y1 = cvRound(y0 + 10000*(a));
            x2 = cvRound(x0 - 10000*(-b));
            y2 = cvRound(y0 - 10000*(a));

            float dist = abs((x2-x1)*(y1-y) - (x1-x)*(y2-y1))/sqrt(pow(x2-x1, 2) + pow(y2-y1, 2));
            distances.push_back(dist);
        }
        if (*min_element(distances.begin(), distances.end()) <= eps_dist_lines){
            circle(cdst, Point(x, y), 1, Scalar(255, 0, 0), 2);
            circle(cdst, Point(x, y), 15 / 2, Scalar(255, 0, 0), 1);
        }
        else{
            circle(cdst, Point(x, y), 1, Scalar(0, 255, 0), 2);
            circle(cdst, Point(x, y), 15 / 2, Scalar(0, 255, 0), 1);
        }
		circle(cdstP, Point(x, y), 1, Scalar(255, 0, 0), 2);
		circle(cdstP, Point(x, y), 15 / 2, Scalar(255, 0, 0), 1);
	}


    // Show results
    imshow("Source", src);
    imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);
    imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
    
    
    

    
    // Wait and Exit
    waitKey(0);
    return 0;
}