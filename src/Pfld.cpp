/*
 * @Author: hzq
 * @Date: 2021-03-18 13:44:39
 * @LastEditTime: 2021-03-22 17:20:36
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Face_Align_PFLD_ncnn/Pfld.cpp
 */


#include "Pfld.hpp"
#include "pfld_lite_opt.id.h"
#include "mat.h"
#include "cpu.h"
static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
static int max_threads_num =4;
inline float exp_special(float x)
{
    int gate = 1;
    if (abs(x) < gate)
        return x * exp(gate);

    if(x > 0)
        return exp(x);
    else
        return -exp(-x);

}

Pfld::Pfld( const std::string &param_bin_path, const std::string &bin_path) {
    ncnn::Option opt;
    opt.lightmode = true;
    max_threads_num = ncnn::get_cpu_count();
    opt.num_threads = max_threads_num;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    pfldnet = new ncnn::Net;
    pfldnet->opt = opt;

    int res = pfldnet->load_param_bin(param_bin_path.data());
    if (res != 0)
    {
        printf("load pfld_lite_opt.param.bin failed!\n");
        return;
    }
    res = pfldnet->load_model(bin_path.data());
    if (res != 0)
    {
        printf("load pfld_lite_opt.bin failed!\n");
        return;
    }
}

Pfld::~Pfld() {
    if( NULL!=pfldnet){
	    delete pfldnet;
	    pfldnet=NULL;
    }    
}

int Pfld::detect(pfldinput &input, pfldoutput &output, int imgwidth, int imgheight) {
    if (NULL == input.bgrimgdata) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    int l32ImgW = imgwidth;
    int l32ImgH = imgheight;

    int left = input.left;
    int top =  input.top;
    int right =input.right;
    int  bottom=input.bottom;

    int l32FaceW = right - left + 1;
    int l32FaceH = bottom - top + 1;

    int l32FaceMaxSize = std::max(l32FaceW,l32FaceH)*1.1;
    int centerX = (left + right)/2;
    int centerY = (top  + bottom)/2;

    int NewLeft = centerX - l32FaceMaxSize*0.5;
    int dx = std::max(0, -NewLeft);
    NewLeft = NewLeft < 0 ? 0 : NewLeft;

    int NewTop = centerY - l32FaceMaxSize*0.5;
    int dy = std::max(0, -NewTop);
    NewTop = NewTop < 0 ? 0 : NewTop;

    int NewRight = centerX + l32FaceMaxSize*0.5;
    int edx = std::max(0, NewRight - l32ImgW);
    NewRight = NewRight > l32ImgW - 1 ? l32ImgW - 1 : NewRight;

    int NewBottom = centerY + l32FaceMaxSize*0.5;
    int edy = std::max(0, NewBottom - l32ImgH);
    NewBottom = NewBottom > l32ImgH - 1 ? l32ImgH - 1 : NewBottom;

    int NewFaceH = NewBottom - NewTop + 1;
    int NewFaceW = NewRight - NewLeft + 1;

    unsigned char *pu8RoiFace =  input.bgrimgdata + NewTop * l32ImgW * 3 + NewLeft * 3;
    //ncnn::resize_bilinear函数在android实机运行时会崩溃，新版ncnn库已修复！
    //bug解决前采用不补边操作
    //ncnn::Mat resize_in = ncnn::Mat::from_pixels_resize(pu8RoiFace,ncnn::Mat::PIXEL_BGR, NewFaceW, NewFaceH, l32ImgW * 3, 112,112);
    //centerX = (NewLeft + NewRight)/2;
    //centerY = (NewTop  + NewBottom)/2;


    ncnn::Mat in = ncnn::Mat::from_pixels(pu8RoiFace, ncnn::Mat::PIXEL_BGR, NewFaceW, NewFaceH, l32ImgW * 3);
    ncnn::Mat resize_in;
    if(dx>0||dy>0||edx>0||edy>0)//必要时扩边
    {
        ncnn::Mat makeBorad_in;
        ncnn::copy_make_border(in, makeBorad_in, dy, edy, dx, edx, ncnn::BORDER_CONSTANT, 0.f);
        ncnn::resize_bilinear(makeBorad_in,resize_in, 112,112);
    }
    else
    {
        ncnn::resize_bilinear(in,resize_in, 112,112);
    }

    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    resize_in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex_pfld = pfldnet->create_extractor();
    ex_pfld.set_num_threads(max_threads_num);
    ex_pfld.input(pfld_lite_opt_param_id::BLOB_input, resize_in);
    ncnn::Mat LandMark_OUT;
    ex_pfld.extract(pfld_lite_opt_param_id::BLOB_landmark, LandMark_OUT);
    float *pf32LandmarkOut = (float*)LandMark_OUT.data;
    output.numofpoints = LandMark_OUT.w/2;
	for (int  j = 0; j < LandMark_OUT.w/2; j++)
	{
		
		//ptFllPoseOutput->landmarks[0].landmarks[j].x =  exp_special(pf32LandmarkOut[2*j]*4) * NewFaceW/112 + centerX;
		//ptFllPoseOutput->landmarks[0].landmarks[j].y =  exp_special(pf32LandmarkOut[2*j+1]*4) * NewFaceH/112 + centerY;

		
        output.landmarks[j].x =  exp_special(pf32LandmarkOut[2*j]*4) * l32FaceMaxSize/112 + centerX;
        output.landmarks[j].y =  exp_special(pf32LandmarkOut[2*j+1]*4) * l32FaceMaxSize/112 + centerY;
	}



    return 0;
}

