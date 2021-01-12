#include "feature_location.hpp"


cv::Mat create_M_int(cv::Mat cameraMatrix)
{
    cv::Mat M_int = cv::Mat::zeros(3, 4, CV_64F);
    M_int.at<double>(0,0) = cameraMatrix.at<double>(0,0);
    M_int.at<double>(1,1) = cameraMatrix.at<double>(1,1);
    M_int.at<double>(2,2) = cameraMatrix.at<double>(2,2);
    M_int.at<double>(0,2) = cameraMatrix.at<double>(0,2);
    M_int.at<double>(1,2) = cameraMatrix.at<double>(1,2);

    return M_int;
}

cv::Mat create_M_ext(std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs)
{
    cv::Mat M_ext = cv::Mat::eye(4, 4, CV_64F);

    cv::Mat rot = cv::Mat::eye(3, 3, CV_64F);
    Rodrigues(rvecs[0], rot);

    for (int i = 0 ; i < 3 ; i++ )
    {
        for (int j = 0 ;  j < 3 ; j++)
        {
            M_ext.at<double>(i,j) = rot.at<double>(i,j);
        }
        M_ext.at<double>(i,3) = (tvecs[0]).at<double>(i,0);
        
    }


    return M_ext;

}

cv::Mat compute_transition_matrix(cv::Mat M_int, cv::Mat M_ext)
{
    cv::Mat M_trans = cv::Mat::zeros(3,4,CV_64F);
    M_trans = M_int * M_ext;
    return M_trans;
}

cv::Point3f get_camera_position(std::vector<cv::Mat> rvecs, std::vector<cv::Mat> tvecs)
{
    cv::Point3f camera_pos;
    cv::Mat rot(3,3,CV_64F);
    cv::Mat rot_tr(3,3,CV_64F);
    cv::Mat pos_camera(3,1,CV_64F);

    Rodrigues(rvecs[0], rot);
    transpose(rot, rot_tr);

    pos_camera = -rot_tr*tvecs[0];

    camera_pos.x = pos_camera.at<double>(0,0);
    camera_pos.y = pos_camera.at<double>(1,0);
    camera_pos.z = pos_camera.at<double>(2,0);

    return camera_pos;
}

cv::Point3f image_to_grid_plan(cv::Point2f point_image, cv::Mat M_transition)
{
    cv::Point3f point_world;
    cv::Mat mat_image = cv::Mat::zeros(3,1,CV_64F);
    cv::Mat mat_world = cv::Mat::zeros(3,1,CV_64F);

    mat_image.at<double>(0, 0) = point_image.x;
    mat_image.at<double>(1, 0) = point_image.y;
    mat_image.at<double>(2, 0) = 1.0f;

    cv::Mat M_trans_2d = cv::Mat::zeros(3,3,CV_64F);

    for (int i = 0 ; i < 3 ; i++)
    {
        M_trans_2d.at<double>(i,0) = M_transition.at<double>(i,0);
        M_trans_2d.at<double>(i,1) = M_transition.at<double>(i,1);
        M_trans_2d.at<double>(i,2) = M_transition.at<double>(i,3);
    }

    cv::Mat M_trans_2d_inv(3,3,CV_64F);
    M_trans_2d_inv = M_trans_2d.inv();

/*
    for (int i = 0 ; i < 3 ; i++ )
    {
        for (int j = 0 ;  j < 3 ; j++)
        {
            std::cout << M_trans_2d_inv.at<double>(i,j) << "   ";
        }
        std::cout << std::endl;
    }*/

    mat_world = M_trans_2d_inv * mat_image;

    double alpha = mat_world.at<double>(2,0);
    //std::cout << "alpha : " << alpha << std::endl;
    point_world.x = mat_world.at<double>(0,0) / alpha;
    point_world.y = mat_world.at<double>(1,0) / alpha;
    point_world.z = 0.0f;




    return point_world;
}

cv::Point2f find_intersection(cv::Point2f feature_world_2d_1, cv::Point2f cam_proj_1, cv::Point2f feature_world_2d_2, cv::Point2f cam_proj_2)
{
    cv::Point3f intersection;
    
    float x1 = feature_world_2d_1.x;
    float x2 = feature_world_2d_2.x;

    float y1 = feature_world_2d_1.y;
    float y2 = feature_world_2d_2.y;

    float z1 = feature_world_2d_1.z;
    float z2 = feature_world_2d_2.z;

    float alpha1 = cam_proj_1.x - feature_world_2d_1.x;
    float alpha2 = cam_proj_2.x - feature_world_2d_2.x;

    float beta1 = cam_proj_1.y - feature_world_2d_1.y;
    float beta2 = cam_proj_2.y - feature_world_2d_2.y;
    
    float gamma1 = cam_proj_1.z - feature_world_2d_1.z;
    float gamma2 = cam_proj_2.z - feature_world_2d_2.z;

    cv::Mat M(2,2, CV_64F);
    cv::Mat T(2,1, CV_64F);
    cv::Mat X(2,1, CV_64F);

    X.at<double>(0,0) = x2-x1;
    X.at<double>(1,0) = y2-y1;

    M.at<double>(0,0) = alpha1;
    M.at<double>(0,1) = - alpha2;
    M.at<double>(1,0) = bata1;
    M.at<double>(1,1) = - beta2;

    cv::Mat M_inv = M.inv();

    T = M_inv*X;

    if (z1 + gamma1*T.at<double>(0,0) != z2 + gamma2*T.at<double>(1,0))
    {
        std::cout << "No intersection" << std::endl;
        return intersection;
    }

    float t = T.at<double>(0,0);
    intersection.x = x1 + alpha1*t;
    intersection.y = y1 + beta1*t;
    intersection.z = z1 + gamma1*t;

    return 
}

cv::Point3f find_feature_3d_im1_im2(std::vector<cv::Point2f> features_im1, std::vector<cv::Point2f> features_im2, cv::Point3f cam_pos_1, cv::Point3f cam_pos_2, cv::Mat M_transition)
{
    cv::Point2f cam_proj_1(cam_pos_1.x, camp_pos_1.y);
    cv::Point2f cam_proj_2(cam_pos_2.x, camp_pos_2.y);

    cv::Point3f feature_world_1 = image_to_grid_plan(features_im1, M_transition);
    cv::Point3f feature_world_2 = image_to_grid_plan(features_im2, M_transition);

    cv::Point2f feature_world_2d_1(feature_world_1.x, feature_world_1.y);
    cv::Point2f feature_world_2d_2(feature_world_2.x, feature_world_2.y);

    cv::Point2f feature_proj = find_intersection(feature_world_2d_1, cam_proj_1, feature_world_2d_2, cam_proj_2);

}