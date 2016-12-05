//
//  test_RFR.cpp
//  RandomFeatureRegistration
//
//  Created by Lili  on 11/27/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#include "test_RFR.h"
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv/cv.h>
#include <opencv2/calib3d/calib3d.hpp>


using namespace std;
using namespace cv;

#define MAX_FRAME 100//1000
#define MIN_NUM_FEAT 2000




    
void test_RandomFeatureMatching()
{
    
    const char* rgb_img_file1 = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.color.png";
    
    const char* depth_img_file1 = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.depth.png";
    
    
    const char* dataset_param_filename="/Users/jimmy/Desktop/PR2017/7scenes_param.txt";
    
    const char* tree_param_filename="/Users/jimmy/Desktop/PR2017/RF_param.txt";
    
    cv::Mat rgb_img1, rgb_img2;
   
    CvxIO::imread_rgb_8u(rgb_img_file1, rgb_img1);
    
    cv::Mat camera_depth_img1, camera_depth_img2;
    bool is_read1 = CvxIO::imread_depth_16bit_to_64f(depth_img_file1, camera_depth_img1);
   
    assert(is_read1);
   
    
    RFR_DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    
    
    RFRTreeParameter tree_param;
    
    tree_param.readFromFile(tree_param_filename);
    
    int num_sample_source = 1000;

    
    

    bool use_depth = true;
    
   
    vector<RFRSourceSample> source_samples = RFRUtil::randomSampleFromRgbdImages(rgb_img_file1,
                                                                                 depth_img_file1,
                                                                                 num_sample_source,
                                                                                 0,
                                                                                 depth_factor,
                                                                                 min_depth,
                                                                                 max_depth,
                                                                                 use_depth,
                                                                                 false);
    
    
    RFRTree tree;
    
    vector<unsigned int> indices;
    
    for(int i = 0; i<source_samples.size(); i++)
    {
        indices.push_back(i);
    }
    
    
    vector<cv::Mat> training_rgbImages;
    training_rgbImages.push_back(rgb_img1);
    tree.buildTree(source_samples, indices, training_rgbImages, tree_param);

    
    
    
    
    std::cout<<"hello4, no problem in buildTree"<<std::endl;
    
    char rgb_img_file2[]   = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000015.color.png";
    char depth_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000015.depth.png";
    CvxIO::imread_rgb_8u(rgb_img_file2, rgb_img2);
    
    bool is_read2 = CvxIO::imread_depth_16bit_to_64f(depth_img_file2, camera_depth_img2);
    assert(is_read2);

   
    // test
    int num_sample_target = 100;
    vector<RFRSourceSample> target_samples = RFRUtil::randomSampleFromRgbdImages(rgb_img_file2,
                                                                                 depth_img_file2,
                                                                                 num_sample_target,
                                                                                 1,
                                                                                 depth_factor,
                                                                                 min_depth,
                                                                                 max_depth,
                                                                                 use_depth,
                                                                                 true);

    vector<RFRTargetSample> predictions;
    for(int i=0; i<target_samples.size(); i++)
    {
        RFRTargetSample pred;
        pred.p2d_ = target_samples[i].p2d_;
        pred.inv_depth_ = target_samples[i].inv_depth_;
        bool is_predict = tree.search(target_samples[i], rgb_img2, pred);
        
        if(is_predict)
        {
            predictions.push_back(pred);            
        }
    }
    
    
    cv::hconcat(rgb_img1,rgb_img2,rgb_img1);
    
     std::vector<cv::Point2f> features_prev, features_next;
    
    for(int i=0; i<predictions.size(); i++)
    {
        cv::circle(rgb_img1, predictions[i].matched_p2d_, 2, cv::Scalar(250,0,250), -1);
        cv::circle(rgb_img1, predictions[i].p2d_+cv::Point2i(rgb_img2.cols, 0),2,cv::Scalar(250,0,250), -1 );
        
        cv::line(rgb_img1, predictions[i].matched_p2d_, predictions[i].p2d_+cv::Point2i(rgb_img2.cols, 0), cv::Scalar(0, 255, 0), 1, 8, 0);
        
        features_prev.push_back(predictions[i].matched_p2d_);
        features_next.push_back(predictions[i].p2d_);
    //    cout<<"matched location "<<predictions[i].matched_p2d_<<endl;
   //     cout<<"original location "<<predictions[i].p2d_<<endl;
      //  double dif_x = predictions[i].matched_p2d_.x-predictions[i].p2d_.x;
      //  double dif_y = predictions[i].matched_p2d_.y-predictions[i].p2d_.y;
     //   printf("the difference between two points dif_x %f, dif_y %f\n", dif_x, dif_y);
    }
  

    std::vector<uchar> status;
    std::vector<float> err;
    
    features_prev.push_back(cv::Point(4, 5));
    cv::calcOpticalFlowPyrLK(rgb_img1, rgb_img2, features_prev, features_next, status, err);
    
    for(int i=0; i<err.size(); i++)
    {
        cout<<err[i]<<endl;
    }
    

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    
    cv::imshow( "Display window", rgb_img1 );                   // Show our image inside it.
    
    cv::waitKey(0);                                          // Wait for a keystroke in the window

}

void test_KLT_tracking()
{
    
    char rgb_img_file1[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.color.png";
    char rgb_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000001.color.png";
    
    cv::Mat rgb_img1 = cv::imread( rgb_img_file1,0 );
    cv::Mat rgb_img2 = cv::imread( rgb_img_file2,0 );
    
    std::vector< cv::Point2f > corners_prev, corners_cur;
    
    // maxCorners – The maximum number of corners to return. If there are more corners
    // than that will be found, the strongest of them will be returned
    int maxCorners = 10;
    
    // qualityLevel – Characterizes the minimal accepted quality of image corners;
    // the value of the parameter is multiplied by the by the best corner quality
    // measure (which is the min eigenvalue, see cornerMinEigenVal() ,
    // or the Harris function response, see cornerHarris() ).
    // The corners, which quality measure is less than the product, will be rejected.
    // For example, if the best corner has the quality measure = 1500,
    // and the qualityLevel=0.01 , then all the corners which quality measure is
    // less than 15 will be rejected.
    double qualityLevel = 0.01;
    
    // minDistance – The minimum possible Euclidean distance between the returned corners
    double minDistance = 20.;
    
    // mask – The optional region of interest. If the image is not empty (then it
    // needs to have the type CV_8UC1 and the same size as image ), it will specify
    // the region in which the corners are detected
    cv::Mat mask;
    
    // blockSize – Size of the averaging block for computing derivative covariation
    // matrix over each pixel neighborhood, see cornerEigenValsAndVecs()
    int blockSize = 3;
    
    // useHarrisDetector – Indicates, whether to use operator or cornerMinEigenVal()
    bool useHarrisDetector = false;
    
    // k – Free parameter of Harris detector
    double k = 0.04;
    
    cv::goodFeaturesToTrack( rgb_img1, corners_prev, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    
    
    std::vector<uchar> status;
    std::vector<float> err;
    
    cv::calcOpticalFlowPyrLK(rgb_img1, rgb_img2, corners_prev, corners_cur, status, err);
    
    for(int i=0; i<err.size(); i++)
    {
        cout<<err[i]<<endl;
    }
    
    Vec3b mycolor(100,0,0);
    
    cv::hconcat(rgb_img1,rgb_img2,rgb_img1);
    
    for( size_t i = 0; i < corners_prev.size(); i++ )
    {
        cv::circle(rgb_img1, corners_prev[i], 4, cv::Scalar( 255. ), -1 );
        
    }
    
    for( size_t i = 0; i< corners_cur.size(); i++)
    {
        cv::circle(rgb_img1,  cv::Point2i(corners_cur[i])+ cv::Point2i(rgb_img2.cols, 0), 4, cv::Scalar( 255. ), -1 );
    }
    
    int min_size=min(corners_prev.size(), corners_cur.size());
    
    for(int i=0; i<min_size; i++)
    {
        cv::line(rgb_img1, cv::Point2i(corners_prev[i]),cv::Point2i(corners_cur[i])+ cv::Point2i(rgb_img2.cols, 0), cv::Scalar(255, 0, 0), 1, 8, 0);
    
    }
    
    
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    
    cv::imshow( "Display window", rgb_img1 );                   // Show our image inside it.
    
    cv::waitKey(0);                                          // Wait for a keystroke in the window
    
    // automatically get rid of points for which 

}

void test_FAST_KLT_VO()
{
    char rgb_filename1[200], rgb_filename2[200];
    
    char depth_filename1[200], depth_filename2[200];
    
    char camera_pose_filename1[200], camera_pose_filename2[200];
    
    /*
    // 4 Scenes
    sprintf(rgb_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/rgb/frame-%06d.color.png", 0);
    sprintf(depth_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.depth.png", 0);
    sprintf(camera_pose_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.pose.txt", 0);
    
    
    sprintf(rgb_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/rgb/frame-%06d.color.png", 20);
    sprintf(depth_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.depth.png", 20);
    sprintf(camera_pose_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.pose.txt", 20);
    */
    
    // 7 Scenes
    sprintf(rgb_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.color.png", 0);
    sprintf(depth_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01//frame-%06d.depth.png", 0);
    sprintf(camera_pose_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01//frame-%06d.pose.txt", 0);
    
    
    sprintf(rgb_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.color.png", 10);
    sprintf(depth_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.depth.png", 10);
    sprintf(camera_pose_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.pose.txt", 10);
    
    const char* dataset_param_filename="/Users/jimmy/Desktop/PR2017/7scenes_param.txt";
    const char* tree_param_filename = "/Users/jimmy/Desktop/PR2017/RF_param.txt";
    
    cv::Mat camera_pose1=Ms7ScenesUtil::read_pose_7_scenes(camera_pose_filename1);
    cv::Mat camera_pose2=Ms7ScenesUtil::read_pose_7_scenes(camera_pose_filename2);
    
   
    
    //read the first two frames from the dataset
    Mat rgb_img1 = imread(rgb_filename1);

    Mat rgb_img2 = imread(rgb_filename2);
    
    cv::Mat depth_img1, depth_img2;
    CvxIO::imread_depth_16bit_to_32f(depth_filename1, depth_img1);
    CvxIO::imread_depth_16bit_to_32f(depth_filename2, depth_img2);
   
    if(!rgb_img1.data || !rgb_img2.data || !depth_img1.data || !depth_img2.data)
    {
        cout<<"--(!) Cannot read images"<<endl;
        return;
    }
    
    Mat gray_img_1, gray_img_2;
   // work with grayscale images
    cvtColor(rgb_img1, gray_img_1, COLOR_BGR2GRAY);
    cvtColor(rgb_img2, gray_img_2, COLOR_BGR2GRAY);
    
    // feature detection, tracking
    vector<Point2f> points1, points2;  //vectors to store the feature points positions
    KLT_Util::FAST_featureDetection(gray_img_1, points1); // detect Good features in img_1
    vector<uchar> status;
    KLT_Util::featureTracking(gray_img_1, gray_img_2, points1, points2, status); //track those features to img_2;
    
    assert(points1.size()==points2.size());
   
    RFR_DatasetParameter dataset_param;
    dataset_param.readFromFileDataParameter(dataset_param_filename);
   
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    
    double min_depth = dataset_param.min_depth_;
    
    double max_depth = dataset_param.max_depth_;
    
    double depth_factor = dataset_param.depth_factor_;
    
    double fx = calibration_matrix.at<double>(0, 0);
    
    double fy = calibration_matrix.at<double>(1, 1);
    
    double cx = calibration_matrix.at<double>(0, 2);
    
    double cy = calibration_matrix.at<double>(1, 2);
    
    
    vector<Point3d>  feature_world3D_1, feature_world3D_2;
    
    cout<<"points1.size() "<<points1.size()<<endl;
    
    vector<cv::Point2d> imgP2d_corresponds_P3d_1, imgP2d_corresponds_P3d_2;
    
    for(int i=0; i<(int)points1.size(); i++)
    {
        auto depthValue1 = depth_img1.at<unsigned short>(points1[i].y, points1[i].x);
        auto depthValue2 = depth_img2.at<unsigned short>(points2[i].y, points2[i].x);
        
        double camera_depth1=depthValue1/(double)depth_factor;
        double camera_depth2=depthValue2/(double)depth_factor;
        
        if(camera_depth1 > min_depth && camera_depth1 < max_depth && camera_depth2 > min_depth && camera_depth2 < max_depth)
        {
            double worldZ1=depthValue1/(double)depth_factor;
            double worldX1=(points1[i].x-cx)*worldZ1/fx;
            double worldY1=(points1[i].y-cy)*worldZ1/fy;
            feature_world3D_1.push_back(Point3d(worldX1,worldY1,worldZ1));
            imgP2d_corresponds_P3d_1.push_back(points1[i]);
            
            double worldZ2=depthValue2/(double)depth_factor;
            double worldX2=(points2[i].x-cx)*worldZ2/fx;
            double worldY2=(points2[i].y-cy)*worldZ2/fy;
            feature_world3D_2.push_back(Point3d(worldX2,worldY2,worldZ2));
            imgP2d_corresponds_P3d_2.push_back(points2[i]);
            
        }
    }
    
    cout<<"feature_world3D_1.size() "<<feature_world3D_1.size()<<" feature_world3D_2.size() "<<feature_world3D_2.size()<<endl;
    
    assert(feature_world3D_1.size()==feature_world3D_2.size());
    
    const double outlier_threshold = 8.0;
    
    cv::Mat estimated_transform, dist_coeff;
    
    CvxPoseEstimation::estimateCameraPose(calibration_matrix,
                                          dist_coeff,
                                          imgP2d_corresponds_P3d_1,
                                          feature_world3D_2,
                                          estimated_transform,
                                          outlier_threshold);
    
    
    
    cv::Mat transformed_pose_gt=camera_pose2*camera_pose1.inv();
    
    double rot_distance, trans_distance;
    
    CvxPoseEstimation::poseDistance(estimated_transform,
                                    transformed_pose_gt,
                                    rot_distance,
                                    trans_distance);
    
    cout<<"rot_distance "<<rot_distance<<endl;
    cout<<"trans_distance "<<trans_distance<<endl;
    
    /*
    cv::hconcat(rgb_img1,rgb_img2,rgb_img1);
    
    for(int i=0; i<points1.size(); i++)
    {
        cv::line(rgb_img1, cv::Point2i(points1[i]),cv::Point2i(points2[i])+ cv::Point2i(rgb_img2.cols, 0), cv::Scalar(255, 0, 0), 1, 8, 0);
        
    }
    
    
    cv::namedWindow( "FAST featureMatching", cv::WINDOW_AUTOSIZE );// Create a window for display.
    
    cv::imshow( "FAST featureMatching", rgb_img1 );                   // Show our image inside it.
    
    cv::waitKey(0);                                          // Wait for a keystroke in the window
    */



}


void test_RandomFeature_KLT()
{
    
    char rgb_filename1[200], rgb_filename2[200];
    
    char depth_filename1[200], depth_filename2[200];
    
    char camera_pose_filename1[200], camera_pose_filename2[200];
    
    /*
     // 4 Scenes
     sprintf(rgb_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/rgb/frame-%06d.color.png", 0);
     sprintf(depth_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.depth.png", 0);
     sprintf(camera_pose_filename1, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.pose.txt", 0);
     
     
     sprintf(rgb_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/rgb/frame-%06d.color.png", 20);
     sprintf(depth_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.depth.png", 20);
     sprintf(camera_pose_filename2, "/Users/jimmy/Desktop/images/4_scenes/apt1/kitchen/data/frame-%06d.pose.txt", 20);
     */
    
    // 7 Scenes
    
    ofstream fout("/Users/jimmy/Desktop/PR2017/transformation_error.txt");
    fout<<"img_index rot_distance  trans_distance"<<endl;
    
    const char* dataset_param_filename="/Users/jimmy/Desktop/PR2017/7scenes_param.txt";
    const char* tree_param_filename = "/Users/jimmy/Desktop/PR2017/RF_param.txt";
    

    for(int i=0; i<100; i++)
    {
    
    sprintf(rgb_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.color.png", i);
    sprintf(depth_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01//frame-%06d.depth.png", i);
    sprintf(camera_pose_filename1, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.pose.txt", i);
    
    
    sprintf(rgb_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.color.png", i+1);
    sprintf(depth_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.depth.png", i+1);
    sprintf(camera_pose_filename2, "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-%06d.pose.txt", i+1);
    
    cv::Mat camera_pose1=Ms7ScenesUtil::read_pose_7_scenes(camera_pose_filename1);
    cv::Mat camera_pose2=Ms7ScenesUtil::read_pose_7_scenes(camera_pose_filename2);
    
    
   cv::Mat estimated_camera_transform;
        
    bool is_camera_pose_estimation=KLT_Util::Random_feature_KLT_VO(rgb_filename1,
                                                                   depth_filename1,
                                                                   rgb_filename2,
                                                                   depth_filename2,
                                                                   dataset_param_filename,
                                                                   tree_param_filename,
                                                                   i,
                                                                   estimated_camera_transform);
        
        
    
    cv::Mat transformed_pose_gt=camera_pose2*camera_pose1.inv();
    
    double rot_distance, trans_distance;
        
   if(is_camera_pose_estimation==true)
   {
    
        CvxPoseEstimation::poseDistance(estimated_camera_transform,
                                        transformed_pose_gt,
                                        rot_distance,
                                        trans_distance);
    
        cout<<"rot_distance "<<rot_distance<<endl;
        cout<<"trans_distance "<<trans_distance<<endl;
    
    
    
        fout<<i<<"  "<<rot_distance<<"  "<<trans_distance<<endl;
    }
    else
    {
        fout<<"wrong pose estimation due to the solvePnPRansac failure from image "<<i<<" to image "<<i+1<<endl;
    }

    }

}
