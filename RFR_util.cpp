//
//  RFR_util.cpp
//  RGBD_RF
//
//  Created by jimmy on 2016-11-26.
//  Copyright (c) 2016 Nowhere Planet. All rights reserved.
//

#include "RFR_util.h"
#include "cvxIO.hpp"

vector<RFRSourceSample>
RFRUtil::randomSampleFromRgbdImages(const char* rgb_img_file,
                                    const char* depth_img_file,
                                    const int num_sample,
                                    const int image_index,
                                    const double depth_factor,
                                    const double min_depth,
                                    const double max_depth,
                                    const bool use_depth,
                                    const bool verbose)
{
    vector<RFRSourceSample> samples;
    assert(rgb_img_file);
    assert(depth_img_file);
    
    cv::Mat camera_depth_img;
    cv::Mat rgb_img;
    
    bool is_read = CvxIO::imread_depth_16bit_to_64f(depth_img_file, camera_depth_img);
    assert(is_read);
    
    CvxIO::imread_rgb_8u(rgb_img_file, rgb_img);
    assert(rgb_img.type()== CV_8UC3);
    
    const int width = rgb_img.cols;
    const int height = rgb_img.rows;
    
    for(int i=0; i<num_sample; i++)
    {
        int x = rand()%width;
        int y = rand()%height;
        
        double camera_depth = camera_depth_img.at<double>(y,x)/depth_factor; // to meter
        //ignore bad depth point
        
        /*
        if(camera_depth<min_depth || max_depth > max_depth)
        {
            continue;
        }*/
        
        double depth = 1.0;
        if(use_depth)
        {
            depth = camera_depth_img.at<double>(y,x)/depth_factor;
        }
        RFRSourceSample sp;
        sp.p2d_ = cv::Vec2i(x,y);
        sp.inv_depth_ = 1.0/depth;
        sp.image_index_ = image_index;
        sp.color_[0] = rgb_img.at<cv::Vec3b>(y,x)[0];
        sp.color_[1] = rgb_img.at<cv::Vec3b>(y,x)[1];
        sp.color_[2] = rgb_img.at<cv::Vec3b>(y,x)[2];
        
        samples.push_back(sp);
    }
    
    if(verbose)
    {
        printf("rgb image is %s\n", rgb_img_file);
        printf("depth image is %s\n", depth_img_file);
        printf("sampled %lu samples\n", samples.size());
    }
    
    return samples;

}

double RFRUtil::inbalance_loss(const int leftNodeSize, const int rightNodeSize)
{
    double dif = leftNodeSize - rightNodeSize;
    double num = leftNodeSize + rightNodeSize;
    double loss = fabs(dif)/num;
    
    assert(loss>=0);
    return loss;

}
