//
//  RFR_util.h
//  RGBD_RF
//
//  Created by jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__RFR_util__
#define __RGBD_RF__RFR_util__

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include <vector>
#include <limits>
#include <unordered_map>


using std::vector;
using std::unordered_map;
using std::string;

// random feature source sample
class RFRSourceSample
{
public:
    cv::Point2i p2d_;   // 2d location
    double inv_depth_;  // inverted gradient, optional in practice
    int image_index_;   // image index
    
    cv::Vec3d color_;   // bgr in OpenCV
    

    RFRSourceSample()
    {
        image_index_ = -1;
        inv_depth_ = 1.0;
    }
    
    cv::Point2i addOffset(const cv::Point2d & offset) const
    {
        int x = cvRound(p2d_.x + offset.x * inv_depth_);
        int y = cvRound(p2d_.y + offset.y * inv_depth_);
        
        return cv::Point2i(x, y);
    }
};


class RFRTargetSample
{
public:
    cv::Point2i p2d_;           // image postion, input
    double inv_depth_;          // inverted gradient, optional in practice;
    
    cv::Vec3d searched_color_;           // bgr in OpenCV
    
    //analysis only
    void *leaf_node_;
    
    RFRTargetSample()
    {
        leaf_node_ = NULL;
    }
    
};

class RFRTreeParameter
{
public:
    bool is_use_depth_;         // true-->use depth, false depth is constant 1.0
    int max_frame_num_;         // sampled frames for a tree
    int sampler_num_per_tree;   // sampler numbers in one frame
    
    int tree_num_;              // number of trees;
    int max_depth_;             // maximum tree depth
    int min_leaf_node_;             // minimum leaf node size    
    
    int max_pixel_offset_;          // int pixel
    int pixel_offset_candidate_num_;    // large number less randomness
    int split_candidate_num_;           // number of split in [v_min, v_max]
    bool verbose_;                      // output training
    bool verbose_leaf_;                 // output leaf information
};

struct RFRSplitParameter
{
    int c1_;                // rgb image channel
    int c2_;
    cv::Point2d offset2_;   // displacement in image [x, y]
    double threshold_;      // threshold of splitting. store result
    
    RFRSplitParameter()
    {
        c1_ = 0;
        c2_ = 0;
        threshold_ = 0.0;
    }
};


class RFRUtil
{
public:
    static vector<RFRSourceSample>
    randomSampleFromRgbdImages(const char* rgb_img_file,
                               const char* depth_img_file,
                               const int   num_sample,
                               const int   image_index,
                               const double depth_factor,
                               const double min_depth,
                               const double max_depth,
                               const bool use_depth,
                               const bool verbose = false);
    
    
    //balance of left and right tree node
    static double inbalance_loss(const int leftNodeSize, const int rightNodeSize);
    

};

class DatasetParameter
{
public:
    double depth_factor_;
    double k_focal_length_x_;
    double k_focal_length_y_;
    double k_camera_centre_u_;
    double k_camera_centre_v_;
    double min_depth_;
    double max_depth_;
    
    DatasetParameter()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 585.0;
        k_focal_length_y_ = 585.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 6.0;
    }
    
    cv::Mat camera_matrix() const
    {
        cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
        K.at<double>(0, 0) = k_focal_length_x_;
        K.at<double>(1, 1) = k_focal_length_y_;
        K.at<double>(0, 2) = k_camera_centre_u_;
        K.at<double>(1, 2) = k_camera_centre_v_;
        
        return K;
    }
    
    void as4Scenes()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 572.0;
        k_focal_length_y_ = 572.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 10.0;
    }
    
    void as7Scenes()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 585.0;
        k_focal_length_y_ = 585.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
        min_depth_ = 0.05;
        max_depth_ = 6.0;
    }
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        const double param_num = 7;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 7);
        
        depth_factor_ = imap[string("depth_factor")];
        k_focal_length_x_ = imap[string("k_focal_length_x")];
        k_focal_length_y_ = imap[string("k_focal_length_y")];
        k_camera_centre_u_ = imap[string("k_camera_centre_u")];
        k_camera_centre_v_ = imap[string("k_camera_centre_v")];
        min_depth_ = imap[string("min_depth")];
        max_depth_ = imap[string("max_depth")];
        return true;
    }
    
    
    bool readFromFileDataParameter(const char* file_name)
    {
        FILE *pf = fopen(file_name, "r");
        if (!pf) {
            printf("Error: can not open %s \n", file_name);
            return false;
        }
        
        const double param_num = 7;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 7);
        fclose(pf);
        
        depth_factor_ = imap[string("depth_factor")];
        k_focal_length_x_ = imap[string("k_focal_length_x")];
        k_focal_length_y_ = imap[string("k_focal_length_y")];
        k_camera_centre_u_ = imap[string("k_camera_centre_u")];
        k_camera_centre_v_ = imap[string("k_camera_centre_v")];
        min_depth_ = imap[string("min_depth")];
        max_depth_ = imap[string("max_depth")];
        
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "depth_factor %lf\n", depth_factor_);
        fprintf(pf, "k_focal_length_x %lf\n", k_focal_length_x_);
        fprintf(pf, "k_focal_length_y %lf\n", k_focal_length_y_);
        
        fprintf(pf, "k_camera_centre_u %lf\n", k_camera_centre_u_);
        fprintf(pf, "k_camera_centre_v %lf\n", k_camera_centre_v_);
        fprintf(pf, "min_depth %f\n", min_depth_);
        fprintf(pf, "max_depth %f\n", max_depth_);
        
        return true;
    }
    
    void printSelf() const
    {
        printf("Dataset parameters:\n");
        printf("depth_factor: %lf\n", depth_factor_);
        printf("k_focal_length_x: %lf\t k_focal_length_y: %lf\n", k_focal_length_x_, k_focal_length_y_);
        printf("k_camera_centre_u: %lf\t k_camera_centre_v_: %lf\n", k_camera_centre_u_, k_camera_centre_v_);
        printf("min depth: %f\t max depth: %f\n", min_depth_, max_depth_);
    }
    
};



#endif /* defined(__RGBD_RF__RFR_util__) */
