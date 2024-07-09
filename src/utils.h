/**
 ******************************************************************************
 * @file           : jj.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-3
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_JJ_H_
#define FACEFUSIONCPP_SRC_JJ_H_

#include <opencv2/opencv.hpp>
#include "typing.h"

namespace Ffc {
using namespace Typing;

float GetIoU(const BoundingBox box1, const BoundingBox box2);
std::vector<int> nms(std::vector<BoundingBox> boxes, std::vector<float> confidences, const float nms_thresh);
cv::Mat create_static_box_mask(const int *crop_size, const float face_mask_blur, const int *face_mask_padding);
cv::Mat paste_back(cv::Mat temp_vision_frame, cv::Mat crop_vision_frame, cv::Mat crop_mask, cv::Mat affine_matrix);
cv::Mat blend_frame(cv::Mat temp_vision_frame, cv::Mat paste_vision_frame, const int FACE_ENHANCER_BLEND = 80);
} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_JJ_H_
