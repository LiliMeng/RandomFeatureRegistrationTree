//
//  RFRTree.h
//  RGBD_RF
//
//  Created by jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#ifndef __RGBD_RF__RFRTree__
#define __RGBD_RF__RFRTree__

// random feature registration

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <vector>
#include "RFRTreeNode.h"
#include "RFR_util.h"
#include "cvxUtil.hpp"

using std::vector;



class RFRTree
{
    RFRTreeNode *root_;
    cv::RNG rng_;  // random number generator
    
    RFRTreeParameter tree_param_;
    

public:
    RFRTree();
    ~RFRTree();
    
    // training random forest by building a decision tree
    // samples: sampled image pixel locations
    // indices: index of samples
    // rgbImages: same size, rgb, 8bit image
    
    bool buildTree(const vector<RFRSourceSample> & samples,
                   const vector<unsigned int> & indices,
                   const vector<cv::Mat> & rgbImages,
                   const RFRTreeParameter & param);
    
    bool search(const RFRSourceSample & sample,
                const cv::Mat & rgbImage,
                RFRTargetSample & predict) const;

private:
    bool configureNode(const vector<RFRSourceSample> & samples,
                       const vector<cv::Mat> & rgbImages,
                       const vector<unsigned int> & indices,
                       int depth,
                       RFRTreeNode *node);
    
    // optimize random feature split parameter
    double optimizeRandomFeature(const vector<RFRSourceSample> & samples,
                                 const vector<cv::Mat> & rgbImages,
                                 const vector<unsigned int> & indices,
                                 const RFRTreeParameter &tree_param,
                                 vector<unsigned int> & left_indices,
                                 vector<unsigned int> & right_indices,
                                 RFRSplitParameter & split_param);
    
    static double bestSplitRandomParameter(const vector<RFRSourceSample> & samples,
                                    const vector<cv::Mat> & rgbImages,
                                    const vector<unsigned int> & indices,
                                    const RFRTreeParameter & tree_param,
                                    RFRSplitParameter & split_param,
                                    vector<unsigned int> & left_indices,
                                    vector<unsigned int> & right_indices);
    
    
    bool setLeafNode(const vector<RFRSourceSample> & samples,
                     const vector<unsigned int>  & indices,
                     int depth,
                     RFRTreeNode* node);
    
    // loss_type: 0-->spatial variance loss, 1-->balance loss
    static double splitLoss(const vector<RFRSourceSample> & samples,
                            const vector<cv::Mat> &rgbImages,
                            const vector<unsigned int> & indices,
                            const RFRSplitParameter & split_param,                           
                            vector<unsigned int> & left_index,
                            vector<unsigned int> & right_index);
    
    
   
    
  
    
    
    bool search(const RFRTreeNode * const node,
                const RFRSourceSample & sample,
                const cv::Mat & rgbImage,
                RFRTargetSample & predict) const;    

   
};


#endif /* defined(__RGBD_RF__RFRTree__) */
