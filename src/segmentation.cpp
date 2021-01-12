#include "segmentation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "feature_location.hpp"


void segmentation(const cv::Mat &im_to_segment, const cv::Mat &M, cv::Mat &segmented){
    segmented = cv::Mat(cv::Size(im_to_segment.cols, im_to_segment.rows), CV_64FC3, cv::Scalar(1, 1, 1));
    cv::Mat grid = cv::imread("data/grid/grid.png", cv::IMREAD_COLOR);
    grid.convertTo(grid, CV_32FC3, 1.0 / 255.0);

    for(int i = 0; i < im_to_segment.rows; ++i){
        std::cout << i << std::endl;
        for(int j = 0; j < im_to_segment.cols; ++j){
            cv::Point2f P_screen(j, i);
            cv::Point3f P_object_mm = image_to_grid_plan(P_screen, M);
            cv::Point3f P_object_px = P_object_mm * 50.0f / 12.4f;
            //std::cout << P_object_px.x << std::endl;
            if(P_object_px.x >= 0 && P_object_px.x < 804 && P_object_px.y > 0 && P_object_px.y < 804){
                //std::cout << "test" << std::endl;
                segmented.at<double>(i, j) = 0; //grid.at<double>(P_object_px.x, P_object_px.y);
            }
        }
    }

    
    cv::Point2f pt_8_1_image(512, 251);
    cv::Point2f pt_15_8_image(613, 589);

    cv::Point3f pt_15_8_world = image_to_grid_plan(pt_15_8_image, M);

    std::cout << "Point X v2 : " << pt_15_8_world.x/12.4f << std::endl;
    std::cout << "Point Y v2 : " << pt_15_8_world.y/12.4f << std::endl;
    std::cout << "Point Z v2 : " << pt_15_8_world.z << std::endl;
    

}
