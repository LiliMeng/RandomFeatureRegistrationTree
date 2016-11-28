//
//  test_RFR.cpp
//  RandomFeatureRegistration
//
//  Created by jimmy on 11/27/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#include "test_RFR.h"


void test_RandomFeatureMatching()
{
    
    char rgb_img_file1[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.color.png";
    char rgb_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000001.color.png";
    
    char depth_img_file1[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.depth.png";
    char depth_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000001.depth.png";
    
    
    const char* dataset_param_filename="/Users/jimmy/Desktop/PR2017/4scenes_param.txt";
    
    cv::Mat rgb_img1, rgb_img2;
   
    CvxIO::imread_rgb_8u(rgb_img_file1, rgb_img1);
    CvxIO::imread_rgb_8u(rgb_img_file2, rgb_img2);
    
    cv::Mat camera_depth_img1, camera_depth_img2;
    bool is_read1 = CvxIO::imread_depth_16bit_to_64f(depth_img_file1, camera_depth_img1);
    bool is_read2 = CvxIO::imread_depth_16bit_to_64f(depth_img_file2, camera_depth_img2);
    assert(is_read1);
    assert(is_read2);
    
    DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    
    int num_sample_source = 5000;
    int num_sample_target = 5000;
    int image_index = 0;

    bool use_depth = 0;
    
    vector<RFRSourceSample> source_samples = RFRUtil::randomSampleFromRgbdImages(rgb_img_file1,
                                                                                 depth_img_file1,
                                                                                 num_sample_source,
                                                                                 image_index,
                                                                                 depth_factor,
                                                                                 min_depth,
                                                                                 max_depth,
                                                                                 use_depth,
                                                                                 false);
    
    
    RFRTree tree;
    RFRTreeParameter param;
    param.verbose_ = true;
    param.min_leaf_node_ = 1;
    param.max_depth_ = 15;
    
    vector<unsigned int> indices;
    
    for(int i = 0; i<num_sample_source; i++)
    {
        indices.push_back(i);
    }
    
    vector<cv::Mat> rgbImages;
    
    rgbImages.push_back(rgb_img1);
    
    tree.buildTree(source_samples, indices, rgbImages, param);

    vector<RFRTargetSample> predictions;
    vector<cv::Point2d> distance;
    
    vector<cv::Point2d> source_points, target_points;
    
    for(int i=0; i<num_sample_target; i++)
    {
        RFRTargetSample pred;
        bool is_predict = tree.search(source_samples[i], rgb_img1, pred);
        
        if(is_predict)
        {
            source_points.push_back(source_samples[i].p2d_);
            target_points.push_back(pred.p2d_);
            predictions.push_back(pred);
        }
    }
    
    cv::Mat matches;
    
    const int sample_num_match = 50;
    
    CvDraw::draw_match_vertical(rgb_img1,
                                rgb_img2,
                                source_points,
                                target_points,
                                matches,
                                sample_num_match);
  
    
}