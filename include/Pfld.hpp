/*
 * @Author: hzq
 * @Date: 2021-03-18 12:40:02
 * @LastEditTime: 2021-03-18 15:29:08
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Face_Align_PFLD_ncnn/Pfld.hpp
 */

#ifndef Pfld_hpp
#define Pfld_hpp

#pragma once

#include "net.h"
#include <algorithm>
#include <iostream>
#include <string>

typedef struct PfldOInput {
    unsigned char *bgrimgdata;
    int left;
    int top;
    int right;
    int bottom;
} pfldinput;

typedef struct Point{
    int x;
    int y;
} point;
typedef struct PfldOutput {

    point *landmarks;
    int numofpoints;
} pfldoutput;

class Pfld {
public:
    Pfld(const std::string &param_bin_path, const  std::string &bin_path);

    ~Pfld();

    int detect(pfldinput &input, pfldoutput &output,  int imgwidth, int imgheight);

private:

    ncnn::Net *pfldnet = NULL;

};

#endif /* Pfld_hpp */
