//
//  RFRTreeNode.cpp
//  RGBD_RF
//
//  Created by Lili & jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "RFRTreeNode.h"


RFRTreeNode::~RFRTreeNode()
{
    if(left_child_)
    {
        delete left_child_;
        left_child_ = NULL;
    }
    
    if(right_child_)
    {
        delete right_child_;
        right_child_ = NULL;
    }

}

RFRTreeNode::RFRTreeNode(const RFRTreeNode & other)
{
    if(&other == this)
    {
        return;
    }
    
    left_child_  = other.left_child_;
    right_child_ = other.right_child_;
    depth_       = other.depth_;
    sample_num_  = other.sample_num_;
    sample_percentage_ = other.sample_percentage_;
    loss_        = other.loss_;
    
}

RFRTreeNode & RFRTreeNode::operator = (const RFRTreeNode & other)
{
    if(&other == this)
    {
        return *this;
    }
    
    left_child_  = other.left_child_;
    right_child_ = other.right_child_;
    depth_       = other.depth_;
    sample_num_  = other.sample_num_;
    sample_percentage_ = other.sample_percentage_;
    loss_        = other.loss_;
    
    return *this;
}

static void write_RFR_prediction(FILE *pf, RFRTreeNode* node)
{
    if(node==NULL)
    {
        fprintf(pf, "#\n");
        return;
    }
    
    //write split node
    RFRSplitParameter param = node->split_param_;
    fprintf(pf,"%2d %d %6d %3.2f %d\t %12.6f %f\t 12.6f\t",
            node->depth_, (int)node->is_leaf_, node->sample_num_, node->sample_percentage_, param.c1_,
            param.offset2_.x, param.offset2_.y, param.c2_,
            param.threshold_);
    
    //write leaf node
    fprintf(pf, "6.3f %6.3d %6.3d\t %6.3f %6.3f %6.3f\t %6.1f %6.1f %6.1f %6.1f\n",
            node->p2d_.x, node->p2d_.y, node->color_mu_[0], node->color_mu_[1], node->color_mu_[2],
            node->color_sigma_[0], node->color_sigma_[1], node->color_sigma_[2]);
   
    write_RFR_prediction(pf, node->left_child_);
    write_RFR_prediction(pf, node->right_child_);

}

/*

bool RFRTreeNode::writeTree(const char* fileName, RFRTreeNode *root)
{
    assert(root);
    FILE *pf = fopen(fileName, "w");
    if(!pf)
    {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    fprintf(pf, "depth\t isLeaf\t sample_num\t sample_percentage\t c1\t displace2\t c2\t threshold\t image_2d \t mean_color\n");
    
    write_RFR_prediction(pf, root);
    fclose(pf);
    return true;

}

static void read_RFR_prediction(FILE *pf, RFRTreeNode* &node)
{
    char lineBuf[1024] = {NULL};
    char *ret = fgets(lineBuf, sizeof(lineBuf), pf);
    
    if(!ret)
    {
        node=NULL;
        return;
    }
    
    if(lineBuf[0]=='#')
    {
        //empty node
        node = NULL;
        return;
    }
    
    // read node parameters
    node = new RFRTreeNode();
    assert(node);
    int depth = 0;
    int isLeaf = 0;
    int sample_num =0;
    double sample_percentage = 0.0;
    
    double d2[2] = {0.0};
    int c1 = 0;  //rgb image channel
    int c2 = 0;
    double threshold = 0;
    double xyz[3] = {0.0};
    double sigma_xyz[3] = {0.0};
    double color[3] = {0.0};
    double color_sigma[3] = {0.0};
    
    int ret_num = sscanf(lineBuf, "%d %d %d %lf %d %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                         &depth, &isLeaf, &sample_num, &sample_percentage,
                         &c1,
                         &d2[0], &d2[1], &c2,
                         &threshold,
                         &xy[0], &xy[1],
                         &sigma_xy[0], &sigma_xy[1], &sigma_xy[2],
                         &color[0], &color[1], &color[2],
                         &color_sigma[0], &color_sigma[1], &color_sigma[2]);
    
    assert(ret_num=19);
    
    node->depth_ = depth;
    node->is_leaf_ = (isLeaf==1);
    node->p2d_ = cv::Point2d(xy[0], xy[1]);
    node->stddev_ = cv::Vec2d(sigma_xy[0], sigma_xy[1]);
    node->color_mu_ = cv::Vec3d(color);
    node->color_sigma_ =cv::Vec3d(color_sigma);
    node->sample_num_ = sample_num;
    node->sample_percentage_ = sample_percentage;
    
    RFRSplitParameter param;
    param.offset2_ = cv::Point2d(d2[0],d2[1]);
    param.c1_ = c1;
    param.c2_ = c2;
    param.threshold_ = threshold;
    
    node->split_param_ = param;
    node->left_child_  = NULL;
    node->right_child_ = NULL;
    
    read_RFR_prediction(pf, node->left_child_);
    read_RFR_prediction(pf, node->right_child_);
}
bool RFRTreeNode::readTree(const char* fileName, RFRTreeNode* & root)
{
    FILE *pf = fopen(fileName, "r");
    if(!pf)
    {
        printf("can not open file %s\n", fileName);
        return false;
    }
    
    //read first line
    char line_buf[1024] = {NULL};
    fgets(line_buf, sizeof(line_buf), pf);
    printf("%s\n", line_buf);
    read_RFR_prediction(pf, root);
    fclose(pf);
    return true;
}
*/
