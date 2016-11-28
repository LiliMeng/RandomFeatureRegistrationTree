//
//  RFRTree.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "RFRTree.h"


static vector<double> random_number_from_range(double min_val, double max_val, int rnd_num)
{
    assert(rnd_num > 0);
    
    cv::RNG rng;
    vector<double> data;
    for(int i=0; i<rnd_num; i++)
    {
        data.push_back(rng.uniform(min_val, max_val));
    }
    return data;
}

RFRTree::RFRTree()
{
    root_ = NULL;
}

RFRTree::~RFRTree()
{
    if(root_)
    {
        delete root_;
        root_ = NULL;
    }
}

bool RFRTree::buildTree(const vector<RFRSourceSample> & samples,
               const vector<unsigned int> & indices,
               const vector<cv::Mat> & rgbImages,
               const RFRTreeParameter & param) 
{
    
    assert(indices.size() <= samples.size());
    root_ = new RFRTreeNode(0);
    
    // set random number
    rng_ = cv::RNG(std::time(0)+10000);
    tree_param_ = param;
    return this->configureNode(samples, rgbImages, indices, 0, root_);

}

double RFRTree::splitLoss(const vector<RFRSourceSample> & samples,
                        const vector<cv::Mat> &rgbImages,
                        const vector<unsigned int> & indices,
                        const RFRSplitParameter & split_param,
                        vector<unsigned int> & left_index,
                        vector<unsigned int> & right_index)
{
    //calculate pixel difference
    vector<double> feature_values(indices.size(), 0.0); //0.0 for invalid pixels
    const int c1 = split_param.c1_;
    const int c2 = split_param.c2_;
    
    for(int i=0; i<indices.size(); i++)
    {
        int index = indices[i];
        assert(index >=0 && index < samples.size());
        RFRSourceSample smp = samples[index];
        cv::Point2i p1 = smp.p2d_;
        cv::Point2i p2 = smp.addOffset(split_param.offset2_);
        
        const cv::Mat rgb_image = rgbImages[smp.image_index_];
        
        bool is_inside_image2 = CvxUtil::isInside(rgb_image.cols, rgb_image.rows, p2.x, p2.y);
        
        double pixel_1_c = 0.0; // out of image as black pixels, random pixel values
        double pixel_2_c = 0.0;
        
        cv::Vec3b pix_1 = rgb_image.at<cv::Vec3b>(p1.y,p1.x); //(row, col)
        pixel_1_c = pix_1[c1];
        
        if(is_inside_image2)
        {
            cv::Vec3b pixel_2 = rgb_image.at<cv::Vec3b>(p2.y, p2.x);
            pixel_2_c = pixel_2[c2];

        }
        
        feature_values[i] = pixel_1_c - pixel_2_c;
    }
    
    double threshold = split_param.threshold_;
    double loss = 0;
    
    for(int j=0; j<feature_values.size(); j++)
    {
        int index = indices[j];
        if(feature_values[j] < threshold)
        {
            left_index.push_back(index);
        }
        else
        {
            right_index.push_back(index);
        }
    }

    assert(left_index.size() + right_index.size() == indices.size());
    
    loss = RFRUtil::inbalance_loss((int)left_index.size(), (int)right_index.size());
    
    return loss;
}




bool RFRTree::configureNode(const vector<RFRSourceSample> & samples,
                   const vector<cv::Mat> & rgbImages,
                   const vector<unsigned int> & indices,
                   int depth,
                   RFRTreeNode *node)
{
    assert(indices.size()<=samples.size());
    
    
    if(depth >=tree_param_.max_depth_ || indices.size() <= tree_param_.min_leaf_node_)
    {
        return this->setLeafNode(samples, indices, depth, node);
    }
    
    //split samples into left and right node using random features
    vector<unsigned int> rnd_left_indices;
    vector<unsigned int> rnd_right_indices;
    
    RFRSplitParameter split_param;
    double min_loss = this->optimizeRandomFeature(samples, rgbImages, indices, tree_param_, rnd_left_indices, rnd_right_indices, split_param);
    
    bool is_split = min_loss < std::numeric_limits<double>::max();
    
    if(is_split)
    {
        node->split_param_ = split_param;
        
        assert(rnd_left_indices.size() + rnd_right_indices.size() == indices.size());
        if(tree_param_.verbose_)
        {
            vector<string> feature_names;
            printf("left, right node number is %lu %lu, percentage: %f loss: %lf\n\n", rnd_left_indices.size(), rnd_right_indices.size(),
                   100.0*rnd_left_indices.size()/indices.size(), min_loss);
        }
        node->sample_num_ = (int)indices.size();
        node->loss_ = RFRUtil::inbalance_loss((int)rnd_left_indices.size(), (int)rnd_right_indices.size());
        
        if(rnd_left_indices.size()!=0)
        {
            RFRTreeNode* left_node = new RFRTreeNode(depth+1);
            this->configureNode(samples, rgbImages, rnd_left_indices, depth+1, left_node);
            left_node->sample_percentage_ = 1.0*rnd_left_indices.size()/indices.size();
            node->left_child_ = left_node;
        }
        
        if(rnd_right_indices.size()!=0)
        {
            RFRTreeNode* right_node = new RFRTreeNode(depth+1);
            this->configureNode(samples, rgbImages, rnd_right_indices, depth+1, right_node);
            right_node->sample_percentage_ = 1.0*rnd_right_indices.size()/indices.size();
            node->right_child_ = right_node;
        
        }
        return true;
    }
    else
    {
        return this->setLeafNode(samples, indices, depth, node);
    }
    
    return true;
}


bool RFRTree::setLeafNode(const vector<RFRSourceSample> & samples,
                 const vector<unsigned int> & indices,
                 int depth,
                 RFRTreeNode* node)
{
    node->depth_ = depth;
    node->is_leaf_ = true;
    node->sample_num_ = (int)indices.size();
    node->loss_ = 0.0;
    
    //maximum sample nodes in the leaf node
    if (indices.size() == 1) {
        node->p2d_ = samples[indices[0]].p2d_;
    }
    else {
        // randomly pick up a sample
        int index = rand()%indices.size();
        node->p2d_ = samples[indices[index]].p2d_;
    }
    
    return true;
}


double RFRTree::optimizeRandomFeature(const vector<RFRSourceSample> & samples,
                             const vector<cv::Mat> & rgbImages,
                             const vector<unsigned int> & indices,
                             const RFRTreeParameter &tree_param,
                             vector<unsigned int> & left_indices,  //output
                             vector<unsigned int> & right_indices,
                             RFRSplitParameter & split_param)
{
    // split samples into left and right node
    const int max_pixel_offset = tree_param.max_pixel_offset_;
    const int max_channel = 3;
    const int max_random_num = tree_param.pixel_offset_candidate_num_;
    
    double min_loss = std::numeric_limits<double>::max();
    for(int i=0; i<max_random_num; i++)
    {
        double x2 = rng_.uniform(-max_pixel_offset, max_pixel_offset);
        double y2 = rng_.uniform(-max_pixel_offset, max_pixel_offset);
        int c1 = rand()%max_channel;
        int c2 = rand()%max_channel;
        
        RFRSplitParameter cur_split_param;
        cur_split_param.offset2_ = cv::Point2d(x2, y2);
        cur_split_param.c1_ = c1;
        cur_split_param.c2_ = c2;
        
        vector<unsigned int> cur_left_indices;
        vector<unsigned int> cur_right_indices;
        double cur_loss = this->bestSplitRandomParameter(samples, rgbImages, indices, tree_param,
                                                         cur_split_param, cur_left_indices, cur_right_indices);
        
        if(cur_loss < min_loss) {
            min_loss = cur_loss;
            left_indices = cur_left_indices;
            right_indices = cur_right_indices;
            split_param = cur_split_param;
        }
            
    }
    return min_loss;


}


double RFRTree::bestSplitRandomParameter(const vector<RFRSourceSample> & samples,
                                         const vector<cv::Mat> & rgbImages,
                                         const vector<unsigned int> & indices,
                                         const RFRTreeParameter & tree_param,
                                         RFRSplitParameter & split_param,
                                         vector<unsigned int> & left_indices,
                                         vector<unsigned int> & right_indices)
{
    double min_loss = std::numeric_limits<double>::max();
    
    const int split_candidate_num = tree_param.split_candidate_num_;    

    // calculate pixel difference
    vector<double> feature_values(indices.size(), 0.0); //0.0 for invalid pixels
    const int c1 = split_param.c1_;
    const int c2 = split_param.c2_;
    
    for(int i=0; i<indices.size(); i++)
    {
        int index = indices[i];
        assert(index >= 0 && index < samples.size());
        RFRSourceSample smp = samples[index];
        cv::Point2i p1 = smp.p2d_;
        cv::Point2i p2 = smp.addOffset(split_param.offset2_);
        
        const cv::Mat rgb_image = rgbImages[smp.image_index_];
        
        bool is_inside_image2 = CvxUtil::isInside(rgb_image.cols, rgb_image.rows, p2.x, p2.y);
        double pixel_1_c = 0.0; // out of image as black pixels, random pixel values
        double pixel_2_c = 0.0;
        
        cv::Vec3b pix_1 = rgb_image.at<cv::Vec3b>(p1.y, p1.x); // (row, col)
        pixel_1_c = pix_1[c1];
        
        if(is_inside_image2)
        {
            cv::Vec3b pixel_2 =rgb_image.at<cv::Vec3b>(p2.y, p2.x);
            pixel_2_c = pixel_2[c2];
        }
        feature_values[i] = pixel_1_c - pixel_2_c;
        
    }
    
    double min_v = *std::min_element(feature_values.begin(), feature_values.end());
    double max_v = *std::max_element(feature_values.begin(), feature_values.end());
    
    vector<double> split_values = random_number_from_range(min_v, max_v, split_candidate_num);
    
    //split data by pixel difference
    bool is_split = false;
    for(int i=0; i<split_values.size(); i++)
    {
        double split_v = split_values[i];
        vector<unsigned int> cur_left_index;
        vector<unsigned int> cur_right_index;
        double cur_loss = 0;
        
        for(int j=0; j<feature_values.size(); j++)
        {
            int index = indices[j];
            if(feature_values[j]<split_v)
            {
                cur_left_index.push_back(index);
            }
            else
            {
                cur_right_index.push_back(index);
            }
        
        }
        
        assert(cur_left_index.size()+cur_right_index.size()==indices.size());
        
        cur_loss = RFRUtil::inbalance_loss((int)cur_left_index.size(), (int)cur_right_index.size());
        
        if(cur_loss < min_loss)
        {
            is_split = true;
            min_loss = cur_loss;
            left_indices = cur_left_index;
            right_indices = cur_right_index;
            split_param.threshold_ = split_v;
        }
        
    }
    if(!is_split)
    {
        return min_loss;
    }
    
    assert(left_indices.size() + right_indices.size() == indices.size());
    
    return min_loss;
    
}


bool RFRTree::search(const RFRSourceSample & sample,
                     const cv::Mat & rgbImage,
                     RFRTargetSample & searchResult) const
{
    assert(root_);
    return this->search(root_, sample, rgbImage, searchResult);

}




bool RFRTree::search(const RFRTreeNode * const node,
                     const RFRSourceSample & sample,
                     const cv::Mat & rgbImage,
                     RFRTargetSample & searchResult) const
{
//    return this->search(node, sample, rgbImage, searchResult);
    // add real code here
    
    return true;

}

