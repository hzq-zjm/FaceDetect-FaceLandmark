/*
 * @Author: hzq
 * @Date: 2021-03-18 13:32:45
 * @LastEditTime: 2021-03-22 17:23:21
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /Face_Align_PFLD_ncnn/main.cpp
 */

#include "UltraFace.hpp"
#include "Pfld.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include<stdio.h>
#include<sys/time.h>

int main() {

    std::string UltraFace_param_bin_path = "./models/slim-Epoch-170_simplified_opt.param.bin";
    std::string UltraFace_bin_path ="./models/slim-Epoch-170_simplified_opt.bin";
    std::string pfld_param_bin_path = "./models/pfld_lite_Epoch-166_opt.param.bin";
    std::string pfld_bin_path="./models/pfld_lite_Epoch-166_opt.bin";
    UltraFace ultraface(UltraFace_param_bin_path, UltraFace_bin_path); 
    Pfld pfld(pfld_param_bin_path,pfld_bin_path);
    
    cv::Mat frame;
    cv::VideoCapture capture("./video/hzq.mp4");
    //cv::VideoCapture capture(0); //开启摄像头
    std::vector<FaceInfo> face_info;
    pfldinput  landmarkinput;
    pfldoutput landmarkoutput;
    landmarkoutput.landmarks =new Point[98];
  
	struct timeval startTime,endTime;
	float Timeuse = 0.f;

    while (true) {
        capture >> frame;         
        if(frame.empty())
            break;    

	    gettimeofday(&startTime,NULL);
        face_info.clear();
        ultraface.detect(frame.data, face_info, frame.cols, frame.rows);
        printf("检测到的人脸数：%d\n", int(face_info.size()));
        landmarkinput.bgrimgdata = frame.data; 
        for (int i = 0; i < int(face_info.size()); i++) {
            auto face = face_info[i];
            cv::Point pt1(face.x1, face.y1);
            cv::Point pt2(face.x2, face.y2);
            cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 1);
            landmarkinput.left = face.x1;
            landmarkinput.right = face.x2;
            landmarkinput.top =face.y1;
            landmarkinput.bottom =face.y2;
            pfld.detect(landmarkinput, landmarkoutput, frame.cols, frame.rows);
            for(int j=0;j<landmarkoutput.numofpoints;j++){
                int x = landmarkoutput.landmarks[j].x;
                int y = landmarkoutput.landmarks[j].y;
	            cv::circle(frame, cv::Point(x,y), 1, cv::Scalar(0, 0, 225), 2,1);
            }
        }
        gettimeofday(&endTime,NULL);
	    Timeuse = 1000000*(endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec);
	    Timeuse /= 1000;
	    printf("Timeuse = %f ms\n",Timeuse);
        cv::imshow("FACE", frame);
        cv::waitKey(20);
    }

    delete[] landmarkoutput.landmarks;
    return 0;
}

