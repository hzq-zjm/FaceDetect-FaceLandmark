#include "UltraFace.hpp"
#include "slim-Epoch-170_simplified_opt.id.h"
#include "cpu.h"
#include "mat.h"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;
static int max_threads_num = 4;
UltraFace::UltraFace(const std::string &param_bin_path,  const std::string &bin_path) {

    ncnn::Option opt;
    opt.lightmode = true;
    max_threads_num = ncnn::get_cpu_count();
    opt.num_threads = max_threads_num;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;


    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
    num_anchors = priors.size();
    /* generate prior anchors finished */
    pultrafacenet = new ncnn::Net;
    pultrafacenet->opt = opt;
    int res = pultrafacenet->load_param_bin(param_bin_path.data());
    if (res != 0)
    {
        printf("load face_detect.param.bin failed!\n");
        return;
    }
    res = pultrafacenet->load_model(bin_path.data());
    if (res != 0)
    {
        printf("load face_detect.bin failed!\n");
        return;
    }

}

UltraFace::~UltraFace() { 
    if(NULL!=pultrafacenet){
        delete pultrafacenet;
        pultrafacenet=NULL;
    }

    for(int i = 0; i <int(priors.size());i++)
        std::vector<float>().swap(priors[i]);
    std::vector<std::vector<float> >().swap(priors);
    std::vector<FaceInfo>().swap(bbox_collection);
}

int UltraFace::detect( unsigned char *imgbgrdata, std::vector<FaceInfo> &face_list, int imgw, int imgh) {
    if (NULL ==imgbgrdata) {
        std::cout << "image is empty ,please check!" << std::endl;
        return -1;
    }

    image_w = imgw;
    image_h = imgh;
    ncnn::Mat ncnn_img = ncnn::Mat::from_pixels_resize(imgbgrdata, ncnn::Mat::PIXEL_BGR2RGB,imgw,imgh, imgw*3, in_w,in_h);

    ncnn_img.substract_mean_normalize(mean_vals, norm_vals);
    bbox_collection.clear();

    ncnn::Extractor ex = pultrafacenet->create_extractor();
    ex.set_num_threads(max_threads_num);
    ex.input(slim_Epoch_170_simplified_opt_param_id::BLOB_input, ncnn_img);
    ncnn::Mat scores, boxes;
    // loc
    ex.extract(slim_Epoch_170_simplified_opt_param_id::BLOB_boxes, boxes);
    // class
    ex.extract(slim_Epoch_170_simplified_opt_param_id::BLOB_scores, scores);
    generateBBox(bbox_collection, scores, boxes, score_threshold, num_anchors);
    nms(bbox_collection, face_list);

    return 0;
}

void UltraFace::generateBBox(std::vector<FaceInfo> &bbox_collection, ncnn::Mat scores, ncnn::Mat boxes, float score_threshold, int num_anchors) {
    for (int i = 0; i < num_anchors; i++) {
        if (scores.channel(0)[i * 2 + 1] > score_threshold) {
            FaceInfo rects;
            float x_center = boxes.channel(0)[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = boxes.channel(0)[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(boxes.channel(0)[i * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(boxes.channel(0)[i * 4 + 3] * size_variance) * priors[i][3];

            rects.x1 = clip(x_center - w / 2.0, 1) * image_w;
            rects.y1 = clip(y_center - h / 2.0, 1) * image_h;
            rects.x2 = clip(x_center + w / 2.0, 1) * image_w;
            rects.y2 = clip(y_center + h / 2.0, 1) * image_h;
            rects.score = clip(scores.channel(0)[i * 2 + 1], 1);
            bbox_collection.push_back(rects);
        }
    }
}

void UltraFace::nms(std::vector<FaceInfo> &input, std::vector<FaceInfo> &output, int type) {
    std::sort(input.begin(), input.end(), [](const FaceInfo &a, const FaceInfo &b) { return a.score > b.score; });   //分数排序

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<FaceInfo> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float nms_score;

            nms_score = inner_area / (area0 + area1 - inner_area);  //交并比

            if (nms_score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        switch (type) {
            case hard_nms: {
                output.push_back(buf[0]);
                break;
            }
            case blending_nms: {
                float total = 0;
                for (int i = 0; i < int(buf.size()); i++) {
                    total += exp(buf[i].score);
                }
                FaceInfo rects;
                memset(&rects, 0, sizeof(rects));
                for (int i = 0; i < int(buf.size()); i++) {
                    float rate = exp(buf[i].score) / total;
                    rects.x1 += buf[i].x1 * rate;
                    rects.y1 += buf[i].y1 * rate;
                    rects.x2 += buf[i].x2 * rate;
                    rects.y2 += buf[i].y2 * rate;
                    rects.score += buf[i].score * rate;
                }
                output.push_back(rects);
                break;
            }
            default: {
                printf("wrong type of nms.");
                exit(-1);
            }
        }
    }
    std::vector<int> ().swap(merged);
}
