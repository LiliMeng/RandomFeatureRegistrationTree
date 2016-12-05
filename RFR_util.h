//
//  RGBGUtil.hpp
//  LoopClosure
//
//  Created by jimmy on 2016-04-02.
//  Copyright © 2016 jimmy. All rights reserved.
//

#ifndef RGBGUtil_cpp
#define RGBGUtil_cpp

#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include <vector>
#include <limits>
#include <unordered_map>

using std::vector;
using std::unordered_map;
using std::string;


class RGBGLearningSample
{
public:
    cv::Point2i p2d_;    // 2d location
    cv::Point3d p3d_;    // 3d coordinate, only used for training, invalid when in testing
    double inv_depth_;  // inverted depth
    int image_index_;      // image index
    
    cv::Vec3d color_;       // bgr in OpenCV
    
public:
    cv::Point2i addOffset(const cv::Point2d & offset) const
    {
        int x = cvRound(p2d_.x + offset.x * inv_depth_);
        int y = cvRound(p2d_.y + offset.y * inv_depth_);
        
        return cv::Point2i(x, y);
    }
};

class RGBGTestingResult
{
public:
    cv::Point2i p2d_;     // image position
    cv::Point3d gt_p3d_;  // as ground truth, not used in prediction
    cv::Point3d predict_p3d_;   // predicted world coordinate
    cv::Point3d predict_error;  // prediction - ground truth
    
    cv::Vec3d  std_;     // prediction standard deviation
    cv::Vec3d sampled_color_;   // image color
    cv::Vec3d predict_color_;   // mean color from leaf node
    
    RGBGTestingResult()
    {
        
    }
    
};


class DatasetParameter
{
public:
    double depth_factor_;
    double k_focal_length_x_;
    double k_focal_length_y_;
    double k_camera_centre_u_;
    double k_camera_centre_v_;
    
    DatasetParameter()
    {
        depth_factor_ = 1000.0;
        k_focal_length_x_ = 585.0;
        k_focal_length_y_ = 585.0;
        k_camera_centre_u_ = 320.0;
        k_camera_centre_v_ = 240.0;
    
    }
    
    bool readFromFileDataParameter(const char* file_name)
    {
        FILE *pf = fopen(file_name, "r");
        assert(pf);
        
        const double param_num = 5;
        unordered_map<std::string, double> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            double val = 0.0;
            int ret = fscanf(pf, "%s %lf", s, &val);
            if (ret!=2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 5);
        
        depth_factor_ = imap[string("depth_factor")];
        k_focal_length_x_ = imap[string("k_focal_length_x")];
        k_focal_length_y_ = imap[string("k_focal_length_y")];
        k_camera_centre_u_ = imap[string("k_camera_centre_u")];
        k_camera_centre_v_ = imap[string("k_camera_centre_v")];
        
        printf("Dataset parameters:\n");
        printf("depth_factor: %lf\n", depth_factor_);
        printf("k_focal_length_x: %lf\t k_focal_length_y: %lf\n", k_focal_length_x_, k_focal_length_y_);
        printf("k_camera_centre_u: %lf\t k_camera_centre_v_: %lf\n", k_camera_centre_u_, k_camera_centre_v_);
        
        fclose(pf);
        
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
       
        return true;
    }
    
    void printSelf() const
    {
        printf("Dataset parameters:\n");
        printf("depth_factor: %lf\n", depth_factor_);
        printf("k_focal_length_x: %lf\t k_focal_length_y: %lf\n", k_focal_length_x_, k_focal_length_y_);
        printf("k_camera_centre_u: %lf\t k_camera_centre_v_: %lf\n", k_camera_centre_u_, k_camera_centre_v_);
       
    }

};

class RGBGTreeParameter
{
public:
    bool is_use_depth_;           // true --> use depth, false depth is constant 1.0
    int max_frame_num_;           // sampled frames for a tree
    int sampler_num_per_frame_;   // sampler numbers in one frame
    
    int tree_num_;                // number of trees
    int max_depth_;
    int min_leaf_node_;
    
    int max_pixel_offset_;            // in pixel
    int pixel_offset_candidate_num_;  // large number less randomness
    int split_candidate_num_;  // number of split in [v_min, v_max]
    int weight_candidate_num_;
    bool verbose_;
    
    
    RGBGTreeParameter()
    {
        // sampler parameters
        is_use_depth_ = false;
        max_frame_num_ = 500;
        sampler_num_per_frame_ = 5000;
        
        // tree structure parameter
        tree_num_ = 5;
        max_depth_ = 15;
        min_leaf_node_ = 50;
        
        // random sample parameter
        max_pixel_offset_ = 131;
        pixel_offset_candidate_num_ = 20;
        split_candidate_num_ = 20;
        weight_candidate_num_ = 10;
        verbose_ = true;
        
       
    }
    
    
    bool readFromFile(FILE *pf)
    {
        assert(pf);
        
        const int param_num = 11;
        unordered_map<std::string, int> imap;
        for(int i = 0; i<param_num; i++)
        {
            char s[1024] = {NULL};
            int val = 0;
            int ret = fscanf(pf, "%s %d", s, &val);
            if (ret != 2) {
                break;
            }
            imap[string(s)] = val;
        }
        assert(imap.size() == 11);
        
        is_use_depth_ = (imap[string("is_use_depth")] == 1);
        max_frame_num_ = imap[string("max_frame_num")];
        sampler_num_per_frame_ = imap[string("sampler_num_per_frame")];
        
        tree_num_ = imap[string("tree_num")];
        max_depth_ = imap[string("max_depth")];
        min_leaf_node_ = imap[string("min_leaf_node")];
        
        max_pixel_offset_ = imap[string("max_pixel_offset")];
        pixel_offset_candidate_num_ = imap[string("pixel_offset_candidate_num")];
        split_candidate_num_ = imap[string("split_candidate_num")];
        
        weight_candidate_num_ = imap[string("weight_candidate_num")];
        verbose_ = imap[string("verbose")];
        
        return true;
    }
    
    bool writeToFile(FILE *pf)const
    {
        assert(pf);
        fprintf(pf, "is_use_depth %d\n", is_use_depth_);
        fprintf(pf, "max_frame_num %d\n", max_frame_num_);
        fprintf(pf, "sampler_num_per_frame %d\n", sampler_num_per_frame_);
        
        fprintf(pf, "tree_num %d\n", tree_num_);
        fprintf(pf, "max_depth %d\n", max_depth_);
        fprintf(pf, "min_leaf_node %d\n", min_leaf_node_);
        
        fprintf(pf, "max_pixel_offset %d\n", max_pixel_offset_);
        fprintf(pf, "pixel_offset_candidate_num %d\n", pixel_offset_candidate_num_);
        fprintf(pf, "split_candidate_num %d\n", split_candidate_num_);
        fprintf(pf, "weight_candidate_num %d\n", weight_candidate_num_);
        fprintf(pf, "verbose %d\n", (int)verbose_);
        return true;
    }
    
    void printSelf() const
    {
        printf("RGB tree parameters:\n");
        printf("max_frame_num: %d\n", max_frame_num_);
        printf("tree_num: %d\t max_depth: %d\t min_leaf_node: %d\n", tree_num_, max_depth_, min_leaf_node_);
        printf("max_pixel_offset: %d\t pixel_offset_candidate_num: %d\t split_candidate_num %d\n",
               max_pixel_offset_,
               pixel_offset_candidate_num_,
               split_candidate_num_);
        printf("weight_candidate_num_: %d\n\n", weight_candidate_num_);
    }
};

struct RGBGTreePruneParameter
{
    double x_max_stddev_;
    double y_max_stddev_;
    double z_max_stddev_;
    
    RGBGTreePruneParameter()
    {
        x_max_stddev_ = std::numeric_limits<double>::max();
        y_max_stddev_ = std::numeric_limits<double>::max();
        z_max_stddev_ = std::numeric_limits<double>::max();
    }
    
    RGBGTreePruneParameter(const double x_stddev,
                           const double y_stddev,
                           const double z_stddev)
    {
        x_max_stddev_ = x_stddev;
        y_max_stddev_ = y_stddev;
        z_max_stddev_ = z_stddev;
    }
};


class RGBGUtil
{
public:
    static void mean_stddev(const vector<RGBGLearningSample> & sample,
                            const vector<unsigned int> & indices,
                            cv::Point3d & mean_pt,
                            cv::Vec3d & stddev);
    
    static void mean_stddev(const vector<cv::Point3d> & points,
                            cv::Point3d & mean_pos,
                            cv::Vec3d & std_pos);
    
    static void mean_stddev(const vector<cv::Vec3d> & data,
                            cv::Vec3d & mean,
                            cv::Vec3d & stddev);
    
    
    
    // spatial variance of selected samples
    static double spatial_variance(const vector<RGBGLearningSample> & samples, const vector<unsigned int> & indices);
    
    // depth image only used to get the ground truth, not used in the feature
    static vector<RGBGLearningSample>
    randomSampleFromRgbdImagesWithoutDepth(const char * rgb_img_file,
                                           const char * depth_img_file,
                                           const char * camera_pose_file,
                                           const int num_sample,
                                           const int image_index,
                                           const bool use_depth = false,
                                           const bool verbose = false);
     static vector<RGBGLearningSample>
    randomSampleFromRgbdImagesWithoutDepth(const char * rgb_img_file,
                                           const char * depth_img_file,
                                           const char * camera_pose_file,
                                           const int num_sample,
                                           const int image_index,
                                           const double depth_factor,
                                           const cv::Mat calibration_matrix,
                                           const double min_depth,
                                           const double max_depth,
                                           const bool use_depth = false,
                                           const bool verbose = false);
   
    
    // depth image only used to get the ground truth, not used in the feature
    // scale: image_size * scale: 0.8, 0.6
    static vector<RGBGLearningSample> randomSampleFromRgbWithScale(const char * rgb_img_file,
                                                                                      const char * depth_img_file,
                                                                                      const char * camera_pose_file,
                                                                                      const int num_sample,
                                                                                      const int image_index,
                                                                                      const double scale,
                                                                                      cv::Mat & scaled_rgb_img);
    
    static cv::Point3d predictionErrorStddev(const vector<RGBGTestingResult> & results);    
    
    
    static vector<double> predictionErrorDistance(const vector<RGBGTestingResult> & results);
    
    static bool readDatasetParameter(const char *file_name, DatasetParameter & dataset_param);
    
    static bool readTreeParameter(const char *file_name, RGBGTreeParameter & tree_param);
    
    static bool readTreePruneParameter(const char *file_name, RGBGTreePruneParameter & param);    
    
};


#endif /* RGBGUtil_cpp */
