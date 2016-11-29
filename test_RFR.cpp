//
//  test_RFR.cpp
//  RandomFeatureRegistration
//
//  Created by Lili on 11/27/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#include "test_RFR.h"


void test_RandomFeatureMatching()
{
    
    char rgb_img_file1[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.color.png";
    char rgb_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000100.color.png";
    
    char depth_img_file1[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000000.depth.png";
    char depth_img_file2[] = "/Users/jimmy/Desktop/images/7_scenes/chess/seq-01/frame-000100.depth.png";
    
    
    const char* dataset_param_filename="/Users/jimmy/Desktop/PR2017/7scenes_param.txt";
    
    const char* tree_param_filename="/Users/jimmy/Desktop/PR2017/RF_param.txt";
    
    cv::Mat rgb_img1, rgb_img2;
   
    CvxIO::imread_rgb_8u(rgb_img_file1, rgb_img1);
    CvxIO::imread_rgb_8u(rgb_img_file2, rgb_img2);
    
    cv::Mat camera_depth_img1, camera_depth_img2;
    bool is_read1 = CvxIO::imread_depth_16bit_to_64f(depth_img_file1, camera_depth_img1);
    bool is_read2 = CvxIO::imread_depth_16bit_to_64f(depth_img_file2, camera_depth_img2);
    assert(is_read1);
    assert(is_read2);
    
    DatasetParameter dataset_param;
    
    std::cout<<"hello1, no problem in dataset_param "<<std::endl;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = cv::Mat::eye(3, 3, CV_64F);
    calibration_matrix.at<double>(0, 0) = dataset_param.k_focal_length_x_;
    calibration_matrix.at<double>(1, 1) = dataset_param.k_focal_length_y_;
    calibration_matrix.at<double>(0, 2) = dataset_param.k_camera_centre_u_;
    calibration_matrix.at<double>(1, 2) = dataset_param.k_camera_centre_v_;
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    
    
    RFRTreeParameter tree_param;
    
    tree_param.readFromFile(tree_param_filename);
    
    
    
    int num_sample_source = tree_param.sampler_num_per_frame_;
    int max_frame_num = tree_param.max_frame_num_;
    int tree_num = tree_param.tree_num_;
    int max_tree_depth = tree_param.max_depth_;
    
    int num_sample_target = 50;
    int image_index = 0;

    bool use_depth = 0;
    
    std::cout<<"hello2, no problem in use_depth = 0"<<std::endl;
    
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
    
    vector<unsigned int> indices;
    
    for(int i = 0; i<num_sample_source; i++)
    {
        indices.push_back(i);
    }
    
    std::cout<<"hello3, no problem in indices.push_bakc(i)"<<std::endl;
    
    vector<cv::Mat> rgbImages;
    
    rgbImages.push_back(rgb_img1);
    
    tree.buildTree(source_samples, indices, rgbImages, tree_param);

    vector<RFRTargetSample> predictions;
    vector<cv::Point2d> distance;
    
    vector<cv::Point2d> source_points, target_points;
    
    std::cout<<"hello4, no problem in buildTree"<<std::endl;
    
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
    
    assert(source_points.size()==target_points.size());
    
    std::cout<<"hello5, no problem in predictions.push_back(pred)"<<std::endl;
    
    cv::Mat matches;
    
    const int sample_num_match = 50;
    
   
    hconcat(rgb_img1,rgb_img2,rgb_img1);
    
    for(int i=0; i<target_points.size(); i++)
    {
        cv::line(rgb_img1, source_points[i], target_points[i]+cv::Point2d(rgb_img2.cols, 0), cv::Scalar(0, 255, 0), 1, 8, 0);
    }
  
    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", rgb_img1 );                   // Show our image inside it.
    
    cv::waitKey(0);                                          // Wait for a keystroke in the window

}
