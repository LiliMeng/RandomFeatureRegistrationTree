//
//  KLT_Util.h
//  RandomFeatureRegistration
//
//  Created by jimmy on 11/30/16.
//  Copyright (c) 2016 UBC. All rights reserved.
//

#ifndef __RandomFeatureRegistration__KLT_Util__
#define __RandomFeatureRegistration__KLT_Util__

#include <iostream>
#include <vector>

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

using namespace cv;
using namespace std;

#include "RFRTreeNode.h"
#include "RFR_util.h"
#include "RFRTree.h"
#include "cvxIO.hpp"
#include "ms7ScenesUtil.hpp"
#include "cvxPoseEstimation.hpp"



class KLT_Util
{
public:
    
    static void FAST_featureDetection(cv::Mat img, vector<Point2f>& points);
    
    static void ORB_featureDetection(cv::Mat img, vector<Point2f>& points);
    
    static void Good_featureDetection(cv::Mat img, vector<Point2f>& points);
    
    static void Random_featureDetection(const char* rgb_img_filename,
                                        const char* depth_img_filename,
                                        const int image_index,
                                        bool use_depth,
                                        const char* dataset_param_filename,
                                        const char* tree_param_filename,
                                        vector<Point2f> &points);
    
    static void Random_featureDetection(cv::Mat img,
                                        cv::Mat depth,
                                        const int image_index,
                                        bool use_depth,
                                        const char* dataset_param_filename,
                                        const char* tree_param_filename,
                                        vector<cv::Point2f> &points);
    
    static void featureTracking(cv::Mat img_1,
                                cv::Mat img_2,
                                 vector<Point2f>& points1,
                                 vector<Point2f>& points2,
                                 vector<uchar>& status);
    
    static void draw_Matches(Mat rgb_img1,
                             Mat rgb_img2,
                             vector<Point2f> points1,
                             vector<Point2f> points2);
    
    static bool Random_feature_KLT_VO(const char* rgb_filename1,
                                      const char* depth_filename1,
                                      const char* rgb_filename2,
                                      const char* depth_filename2,
                                      const char* dataset_param_filename,
                                      const char* tree_param_filename,
                                      int image_index,
                                      Mat &estimated_camera_transform);
    
    

};
    
    
    

#endif /* defined(__RandomFeatureRegistration__KLT_Util__) */

