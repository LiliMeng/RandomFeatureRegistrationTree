//
//  KLT_Util.cpp
//  RandomFeatureRegistration
//
//  Created by jimmy on 11/30/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#include "KLT_Util.h"
#include <iostream>
#include "cvxPoseEstimation.hpp"
#include "RGBGUtil.hpp"


void KLT_Util::FAST_featureDetection(Mat img, vector<Point2f>& points)
{   //uses FAST as of now, modify parameters as necessary
        vector<KeyPoint> keypoints;
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        KeyPoint::convert(keypoints, points, vector<int>());
}

void KLT_Util::ORB_featureDetection(Mat img, vector<Point2f>& points)
{
    int nfeatures=3000;
    cv::Ptr<cv::Feature2D> feature_2d = cv::ORB::create(nfeatures);
    vector<KeyPoint> keypoints;
    //-- Step 1: Detect the keypoints:
    feature_2d->detect( img, keypoints );

    KeyPoint::convert(keypoints, points, vector<int>());
}


void KLT_Util::Good_featureDetection(Mat img, vector<Point2f>& points)
{
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
    
    cv::goodFeaturesToTrack(img, points, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k );
    
}


void KLT_Util::Random_featureDetection(const char* rgb_img_filename,
                                       const char* depth_img_filename,
                                       const int image_index,
                                       bool use_depth,
                                       const char* dataset_param_filename,
                                       const char* tree_param_filename,
                                       vector<Point2f> &points)
{
    cv::Mat rgb_img1;
    
    CvxIO::imread_rgb_8u(rgb_img_filename, rgb_img1);
    
    cv::Mat camera_depth_img1;
    
    bool is_read1 = CvxIO::imread_depth_16bit_to_64f(depth_img_filename, camera_depth_img1);
    
    assert(is_read1);

    RFR_DatasetParameter dataset_param;

    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
   
    
    
    RFRTreeParameter tree_param;
    
    tree_param.readFromFile(tree_param_filename);
    int num_sample_source = tree_param.sampler_num_per_frame_;
    
    cout<<"num_sample per RGB and depth image "<<num_sample_source<<endl;
    
    
    vector<RFRSourceSample> source_samples = RFRUtil::randomSampleFromRgbdImages(rgb_img_filename,
                                                                                 depth_img_filename,
                                                                                 num_sample_source,
                                                                                 image_index,
                                                                                 depth_factor,
                                                                                 min_depth,
                                                                                 max_depth,
                                                                                 use_depth,
                                                                                 false);
    for(int i=0; i<source_samples.size(); i++)
    {
        points.push_back(cv::Point2f(source_samples[i].p2d_));
    }
    
}



void KLT_Util::Random_featureDetection(cv::Mat rgb_img,
                                       cv::Mat depth_img,
                                       const int image_index,
                                       bool use_depth,
                                       const char* dataset_param_filename,
                                       const char* tree_param_filename,
                                       vector<Point2f> &points)
{
    
    RFR_DatasetParameter dataset_param;
    
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    
    
    RFRTreeParameter tree_param;
    
    tree_param.readFromFile(tree_param_filename);
    
    int num_sample_source = 1000;
    
    
    vector<RFRSourceSample> source_samples = RFRUtil::randomSampleFromRgbdImages(rgb_img,
                                                                                 depth_img,
                                                                                 num_sample_source,
                                                                                 image_index,
                                                                                 depth_factor,
                                                                                 min_depth,
                                                                                 max_depth,
                                                                                 use_depth,
                                                                                 false);
    for(int i=0; i<source_samples.size(); i++)
    {
        points.push_back(cv::Point2f(source_samples[i].p2d_));
    }
    
}

void KLT_Util::featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)
{
    //this function automatically gets rid of points for which tracking fails
    
    
    vector<float> err;
    Size winSize=Size(21,21);
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    
    int img2_width = img_2.cols;
    int img2_height = img_2.rows;
    
    
    //getting rid of points for which the KLT tracking failed or those who have gone outside of the frame
    int indexCorrection = 0;
    for(int i = 0; i<status.size(); i++)
    {
        Point2f pt = points2.at(i-indexCorrection);
        
        if(status[i]==0 || pt.x<0 || pt.y<0 || pt.x > img2_width || pt.y >img2_height )
        {
            if(pt.x<0||pt.y<0 || pt.x > img2_width || pt.y >img2_height)
            {
                status[i]=0;
            }
            
            points1.erase(points1.begin() + i - indexCorrection);
            points2.erase(points2.begin() + i - indexCorrection);
            indexCorrection++;
        }
    }
    
}

void KLT_Util::draw_Matches(Mat rgb_img1, Mat rgb_img2,  vector<Point2f> points1, vector<Point2f> points2)
{
    cv::hconcat(rgb_img1,rgb_img2,rgb_img1);
    
    for(int i=0; i<points1.size(); i++)
    {
        cv::line(rgb_img1, cv::Point2i(points1[i]),cv::Point2i(points2[i])+ cv::Point2i(rgb_img2.cols, 0), cv::Scalar(255, 0, 0), 1, 8, 0);
        
    }
    
    
    cv::namedWindow( "FeatureMatching", cv::WINDOW_AUTOSIZE );// Create a window for display.
    
    cv::imshow( "FeatureMatching", rgb_img1 );                   // Show our image inside it.
    
    cv::waitKey(0);                                          // Wait for a keystroke in the window
    
}

bool KLT_Util::Random_feature_KLT_VO(const char* rgb_filename1,
                                     const char* depth_filename1,
                                     const char* rgb_filename2,
                                     const char* depth_filename2,
                                     const char* dataset_param_filename,
                                     const char* tree_param_filename,
                                     int image_index,
                                     Mat &estimated_camera_transform)
{
    
    Mat rgb_img1 = imread(rgb_filename1);
    Mat rgb_img2 = imread(rgb_filename2);
    
    cv::Mat depth_img1, depth_img2;
    CvxIO::imread_depth_16bit_to_32f(depth_filename1, depth_img1);
    CvxIO::imread_depth_16bit_to_32f(depth_filename2, depth_img2);
    
    if(!rgb_img1.data || !rgb_img2.data || !depth_img1.data || !depth_img2.data)
    {
        cout<<"--(!) Cannot read images"<<endl;
        return false;
    }

    RFR_DatasetParameter dataset_param;
    dataset_param.readFromFileDataParameter(dataset_param_filename);
    double min_depth = dataset_param.min_depth_;
    double max_depth = dataset_param.max_depth_;
    double depth_factor = dataset_param.depth_factor_;
    
    cv::Mat calibration_matrix = dataset_param.camera_matrix();
    
    
    double fx = calibration_matrix.at<double>(0, 0);
    double fy = calibration_matrix.at<double>(1, 1);
    
    double cx = calibration_matrix.at<double>(0, 2);
    double cy = calibration_matrix.at<double>(1, 2);
    
    Mat gray_img_1, gray_img_2;
    // work with grayscale images
    cvtColor(rgb_img1, gray_img_1, COLOR_BGR2GRAY);
    cvtColor(rgb_img2, gray_img_2, COLOR_BGR2GRAY);
    
    // feature detection, tracking
    vector<Point2f> points1, points2;  //vectors to store the feature points positions
    
    
    bool use_depth = true;
    
    KLT_Util::Random_featureDetection(rgb_filename1,
                                      depth_filename1,
                                      image_index,
                                      use_depth,
                                      dataset_param_filename,
                                      tree_param_filename,
                                      points1);    // detect RANDOM features in img_1
    
    vector<uchar> status;
    
    
    KLT_Util::featureTracking(rgb_img1, rgb_img2, points1, points2, status); //track those features to img_2;
    
    
    assert(points1.size()==points2.size());
    
    cout<<"points1.size() "<<points1.size()<<endl;

    
    vector<cv::Point2d> imgP2d_corresponds_P3d_1, imgP2d_corresponds_P3d_2;
    vector<Point3d>  feature_world3D_1, feature_world3D_2;
    
    for(int i=0; i<(int)points1.size(); i++)
    {
        auto depthValue1 = depth_img1.at<unsigned short>(points1[i].y, points1[i].x);
       // cout<<"points2[i].y "<<points2[i].y<<" points2[i].x "<<points2[i].x<<endl;
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
    
    bool is_estimate_camPose=CvxPoseEstimation::estimateCameraPose(calibration_matrix,
                                                                   dist_coeff,
                                                                   imgP2d_corresponds_P3d_1,
                                                                   feature_world3D_2,
                                                                   estimated_camera_transform,
                                                                   outlier_threshold);
    
  //  draw_Matches(rgb_img1, rgb_img2, points1, points2);
    
    return is_estimate_camPose;
  
    

}





