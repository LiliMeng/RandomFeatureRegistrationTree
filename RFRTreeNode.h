//
//  RFRTreeNode.h
//  RGBD_RF
//
//  Created by jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__RFRTreeNode__
#define __RGBD_RF__RFRTreeNode__

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <vector>
#include "RFR_util.h"

class RFRTreeNode
{
public:
    RFRTreeNode *left_child_;
    RFRTreeNode *right_child_;
    int depth_;
    bool is_leaf_;
    
    int sample_num_;
    double sample_percentage_;
    double loss_;                       // loss parameter   
    
    RFRSplitParameter split_param_;      // random feature split parameter
    
    cv::Point2i p2d_;       // image coordinate of the node, only available in leaf node
    cv::Vec3d   stddev_;    // standard deviation of node, only in leaf node
    
    // leaf color information
    cv::Vec3d color_mu_;     // mean
    cv::Vec3d color_sigma_;  // standard deviation
 
    
public:
    
    RFRTreeNode(int depth)
    {
        left_child_ = NULL;
        right_child_ = NULL;
        is_leaf_ = false;
        depth_ = depth;
        sample_num_ = 0;
        sample_percentage_ = 0.0;
        loss_ = 0.0;
    }
    ~RFRTreeNode();
    
    RFRTreeNode(const RFRTreeNode &other);
    RFRTreeNode & operator = (const RFRTreeNode &other);
    
    static bool writeTree(const char* fileName, RFRTreeNode* root);
    static bool readTree(const char*  fileName, RFRTreeNode* &root);
};

#endif /* defined(__RGBD_RF__RFRTreeNode__) */
