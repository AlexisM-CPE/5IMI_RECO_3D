#include "segmentation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "feature_location.hpp"


void segmentation(const cv::Mat &im_to_segment, const cv::Mat &M, cv::Mat &segmented){
    segmented = cv::Mat(cv::Size(im_to_segment.cols, im_to_segment.rows), CV_32FC3, cv::Scalar(1, 1, 1));
    cv::Mat segmented_uint = cv::Mat(cv::Size(im_to_segment.cols, im_to_segment.rows), CV_8UC1, cv::Scalar(255));
    cv::Mat mask = cv::Mat(cv::Size(im_to_segment.cols, im_to_segment.rows), CV_8UC1, cv::Scalar(0));
    cv::Mat grid = cv::imread("data/grid/grid.png", cv::IMREAD_COLOR);
    grid.convertTo(grid, CV_32FC3, 1.0 / 255.0);

    cv::Mat im_to_segment_HSV;
    cv::cvtColor(im_to_segment, im_to_segment_HSV, cv::COLOR_BGR2HSV);
    cv::Mat grid_HSV;
    cv::cvtColor(grid, grid_HSV, cv::COLOR_BGR2HSV);

    std::cout << "Segmentation..." << std::endl;
    for(int i = 0; i < im_to_segment.rows; ++i){

        for(int j = 0; j < im_to_segment.cols; ++j){
            cv::Point2f P_screen(j, i);
            cv::Point3f P_object_mm = image_to_grid_plan(P_screen, M);
            cv::Point3f P_object_px = P_object_mm * 50.0f / 12.4f;

            if(P_object_px.x >= 0 && P_object_px.x < 804 && P_object_px.y > 0 && P_object_px.y < 804) {
                cv::Vec3f a = grid.at<cv::Vec3f>(P_object_px.x, P_object_px.y);
                cv::Vec3f b = im_to_segment.at<cv::Vec3f>(i, j);
                cv::Vec3f c = a - b;
                //c[0] /= 360;
                float v = c[0]*c[0]+c[1]*c[1]+c[2]*c[2];
                uint8_t v_u = std::min(int(255*(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])), 255);
                cv::Vec3f d = cv::Vec3f(v, v, v);
                segmented.at<cv::Vec3f>(i, j) = d;
                segmented_uint.at<uint8_t>(i,j) = v_u;
                mask.at<uint8_t>(i,j) = 255;
            }
        }
    }



    cv::Mat final_seg;
    adaptiveThreshold(segmented_uint, final_seg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 1001, -10);
    final_seg = final_seg.mul(mask); // ne garde que la grille

    cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT , cv::Size( 2*5 + 1, 2*5+1 ));
    cv::Mat result_morpho;
    cv::morphologyEx( final_seg, result_morpho, cv::MORPH_CLOSE, element );
    element = cv::getStructuringElement( cv::MORPH_RECT , cv::Size( 2*7 + 1, 2*7+1 ));
    cv::morphologyEx( result_morpho, result_morpho, cv::MORPH_OPEN, element );

    element = cv::getStructuringElement( cv::MORPH_RECT , cv::Size( 2*25 + 1, 2*25+1 ));
    cv::morphologyEx( result_morpho, result_morpho, cv::MORPH_DILATE , element );
    //element = cv::getStructuringElement( cv::MORPH_RECT , cv::Size( 2*50 + 1, 2*50+1 ));
    //cv::morphologyEx( result_morpho, result_morpho, cv::MORPH_CLOSE , element );
    
    imshow("test et match", result_morpho);
}
