/**
 ******************************************************************************
 * @file           : face_helper.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#include <opencv2/opencv.hpp>
#include "face_helper.h"

namespace Ffc {
using namespace Typing;
std::unordered_map<std::string, std::vector<cv::Point2f>> FaceHelper::WarpTemplateMap = {
    {"arcface_112_v1", {{0.35473214, 0.45658929}, {0.64526786, 0.45658929}, {0.50000000, 0.61154464}, {0.37913393, 0.77687500}, {0.62086607, 0.77687500}}},
    {"arcface_112_v2", {{0.34191607, 0.46157411}, {0.65653393, 0.45983393}, {0.50022500, 0.64050536}, {0.37097589, 0.82469196}, {0.63151696, 0.82325089}}},
    {"arcface_128_v2", {{0.36167656, 0.40387734}, {0.63696719, 0.40235469}, {0.50019687, 0.56044219}, {0.38710391, 0.72160547}, {0.61507734, 0.72034453}}},
    {"ffhq_512", {{0.37691676, 0.46864664}, {0.62285697, 0.46912813}, {0.50123859, 0.61331904}, {0.39308822, 0.72541100}, {0.61150205, 0.72490465}}}};

float FaceHelper::GetIoU(const BoundingBox box1, const BoundingBox box2) {
    float x1 = std::max(box1.xmin, box2.xmin);
    float y1 = std::max(box1.ymin, box2.ymin);
    float x2 = std::min(box1.xmax, box2.xmax);
    float y2 = std::min(box1.ymax, box2.ymax);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

std::vector<int> FaceHelper::apply_nms(std::vector<BoundingBox> boxes, std::vector<float> confidences, const float nms_thresh) {
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2) { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i) {
        if (isSuppressed[i]) {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j) {
            if (isSuppressed[j]) {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh) {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++) {
        if (!isSuppressed[i]) {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

std::vector<cv::Point2f> FaceHelper::convertFaceLandmarks68To5(const std::vector<cv::Point2f> &faceLandmarks68) {
    std::vector<cv::Point2f> faceLandmarks5(5);
    float x = 0, y = 0;
    for (int i = 36; i < 42; i++) /// left_eye
    {
        x += faceLandmarks68[i].x;
        y += faceLandmarks68[i].y;
    }
    x /= 6;
    y /= 6;
    faceLandmarks5[0] = cv::Point2f(x, y); /// left_eye

    x = 0, y = 0;
    for (int i = 42; i < 48; i++) /// right_eye
    {
        x += faceLandmarks68[i].x;
        y += faceLandmarks68[i].y;
    }
    x /= 6;
    y /= 6;
    faceLandmarks5[1] = cv::Point2f(x, y); /// right_eye

    faceLandmarks5[2] = faceLandmarks68[30]; /// nose
    faceLandmarks5[3] = faceLandmarks68[48]; /// left_mouth_end
    faceLandmarks5[4] = faceLandmarks68[54]; /// right_mouth_end

    return faceLandmarks5;
}

std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
FaceHelper::warpFaceByFaceLandmarks5(const cv::Mat &tempVisionFrame,
                                     const std::vector<cv::Point2f> &faceLandmark5,
                                     const std::string &warpTemplate,
                                     const cv::Size &cropSize) {
    cv::Mat affineMatrix = estimateMatrixByFaceLandmark5(faceLandmark5, warpTemplate, cropSize);
    Typing::VisionFrame cropVision;
    cv::warpAffine(tempVisionFrame, cropVision, affineMatrix, cropSize, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return std::make_shared<std::tuple<VisionFrame, cv::Mat>>(cropVision, affineMatrix);
}
std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
FaceHelper::warpFaceByFaceLandmarks5(const VisionFrame &tempVisionFrame,
                                     const FaceLandmark &faceLandmark5,
                                     const std::vector<cv::Point2f> &warpTemplate,
                                     const cv::Size &cropSize) {
    cv::Mat affineMatrix = estimateMatrixByFaceLandmark5(faceLandmark5, warpTemplate, cropSize);
    VisionFrame cropVision;
    cv::warpAffine(tempVisionFrame, cropVision, affineMatrix, cropSize, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return std::make_shared<std::tuple<VisionFrame, cv::Mat>>(cropVision, affineMatrix);
}
cv::Mat FaceHelper::estimateMatrixByFaceLandmark5(Typing::FaceLandmark landmark5,
                                                  const std::string &warpTemplate,
                                                  const cv::Size cropSize) {
    std::vector<cv::Point2f> normedWarpTemplate = WarpTemplateMap.at(warpTemplate);
    for (auto &point : normedWarpTemplate) {
        point.x *= (float)cropSize.width;
        point.y *= (float)cropSize.height;
    }
    cv::Mat affineMatrix = cv::estimateAffinePartial2D(landmark5, normedWarpTemplate,
                                                       cv::noArray(), cv::RANSAC, 100);
    return affineMatrix;
}
cv::Mat FaceHelper::estimateMatrixByFaceLandmark5(Typing::FaceLandmark landmark5,
                                                  const std::vector<cv::Point2f> &warpTemplate,
                                                  const cv::Size cropSize) {
    cv::Mat affineMatrix = cv::estimateAffinePartial2D(landmark5, warpTemplate,
                                                       cv::noArray(), cv::RANSAC, 100);
    return affineMatrix;
}

std::shared_ptr<std::tuple<cv::Mat, cv::Mat>>
FaceHelper::warpFaceByTranslation(const cv::Mat &tempVisionFrame,
                                  const std::vector<float> &translation,
                                  const float &scale, const cv::Size &cropSize) {
    cv::Mat affineMatrix = (cv::Mat_<float>(2, 3) << scale, 0.f, translation[0], 0.f, scale, translation[1]);
    cv::Mat cropImg;
    warpAffine(tempVisionFrame, cropImg, affineMatrix, cropSize);
    return std::make_shared<std::tuple<cv::Mat, cv::Mat>>(cropImg, affineMatrix);
}

std::shared_ptr<Typing::FaceLandmark> FaceHelper::convertFaceLandmark68To5(const FaceLandmark &faceLandmark68) {
    Typing::FaceLandmark faceLandmark5_68(5);
    faceLandmark5_68.resize(5);
    float x = 0, y = 0;
    for (int i = 36; i < 42; i++) /// left_eye
    {
        x += faceLandmark68[i].x;
        y += faceLandmark68[i].y;
    }
    x /= 6;
    y /= 6;
    faceLandmark5_68[0] = cv::Point2f(x, y); /// left_eye

    x = 0, y = 0;
    for (int i = 42; i < 48; i++) /// right_eye
    {
        x += faceLandmark68[i].x;
        y += faceLandmark68[i].y;
    }
    x /= 6;
    y /= 6;
    faceLandmark5_68[1] = cv::Point2f(x, y);  /// right_eye
    faceLandmark5_68[2] = faceLandmark68[30]; /// nose
    faceLandmark5_68[3] = faceLandmark68[48]; /// left_mouth_end
    faceLandmark5_68[4] = faceLandmark68[54]; /// right_mouth_end

    return std::make_shared<Typing::FaceLandmark>(std::move(faceLandmark5_68));
}
cv::Mat FaceHelper::pasteBack(cv::Mat temp_vision_frame, cv::Mat crop_vision_frame, cv::Mat crop_mask, cv::Mat affine_matrix) {
    cv::Mat inverse_matrix;
    cv::invertAffineTransform(affine_matrix, inverse_matrix);
    cv::Mat inverse_mask;
    cv::Size temp_size(temp_vision_frame.cols, temp_vision_frame.rows);
    warpAffine(crop_mask, inverse_mask, inverse_matrix, temp_size);
    inverse_mask.setTo(0, inverse_mask < 0);
    inverse_mask.setTo(1, inverse_mask > 1);
    cv::Mat inverse_vision_frame;
    warpAffine(crop_vision_frame, inverse_vision_frame, inverse_matrix, temp_size, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    std::vector<cv::Mat> inverse_vision_frame_bgrs(3);
    split(inverse_vision_frame, inverse_vision_frame_bgrs);
    std::vector<cv::Mat> temp_vision_frame_bgrs(3);
    split(temp_vision_frame, temp_vision_frame_bgrs);
    for (int c = 0; c < 3; c++) {
        inverse_vision_frame_bgrs[c].convertTo(inverse_vision_frame_bgrs[c], CV_32FC1); ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
        temp_vision_frame_bgrs[c].convertTo(temp_vision_frame_bgrs[c], CV_32FC1);       ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
    }
    std::vector<cv::Mat> channel_mats(3);

    channel_mats[0] = inverse_mask.mul(inverse_vision_frame_bgrs[0]) + temp_vision_frame_bgrs[0].mul(1 - inverse_mask);
    channel_mats[1] = inverse_mask.mul(inverse_vision_frame_bgrs[1]) + temp_vision_frame_bgrs[1].mul(1 - inverse_mask);
    channel_mats[2] = inverse_mask.mul(inverse_vision_frame_bgrs[2]) + temp_vision_frame_bgrs[2].mul(1 - inverse_mask);

    cv::Mat paste_vision_frame;
    merge(channel_mats, paste_vision_frame);
    paste_vision_frame.convertTo(paste_vision_frame, CV_8UC3);
    return paste_vision_frame;
}
} // namespace Ffc