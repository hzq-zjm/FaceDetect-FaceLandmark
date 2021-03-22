/*
 * @Author: your name
 * @Date: 2021-03-18 13:32:09
 * @LastEditTime: 2021-03-22 17:19:46
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /Face_Align_PFLD_ncnn/include/UltraFace.hpp
 */
#ifndef UltraFace_hpp
#define UltraFace_hpp

#pragma once

#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#define num_featuremap 4
#define hard_nms 1
#define blending_nms 2 /* mix nms was been proposaled in paper blaze face, aims to minimize the temporal jitter*/

typedef struct FaceInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} FaceInfo;

class UltraFace {
public:
    UltraFace(const std::string &param_bin_path, const std::string &bin_path);

    ~UltraFace();

    int detect( unsigned char *imgbgrdata, std::vector<FaceInfo> &face_list, int imgw, int imgh);

private:
    void generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors);

    void nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type = hard_nms);  //int type = blending_nms

private:
    ncnn::Net *pultrafacenet =NULL;;

    const int num_thread = 4;
    int image_w;
    int image_h;

    const int in_w = 128;
    const int in_h = 96;
    int num_anchors;

 
    const float score_threshold =0.80;
    const float iou_threshold = 0.3;


    const float mean_vals[3] = {127, 127, 127};
    const float norm_vals[3] = {1.0 / 128, 1.0 / 128, 1.0 / 128};

    const float center_variance = 0.1;
    const float size_variance = 0.2;
    const std::vector<std::vector<float>> min_boxes = {
            {10.0f,  16.0f,  24.0f},
            {32.0f,  48.0f},
            {64.0f,  96.0f},
            {128.0f, 192.0f, 256.0f}};
    const std::vector<float> strides = {8.0, 16.0, 32.0, 64.0};
    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    std::vector<int> w_h_list = {in_w, in_h};;

    std::vector<std::vector<float>> priors = {};

    std::vector<FaceInfo> bbox_collection;
};

#endif /* UltraFace_hpp */
