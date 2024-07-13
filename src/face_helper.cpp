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

float FaceHelper::getIoU(const BoundingBox &box1, const BoundingBox &box2) {
    float x1 = std::max(box1.xmin, box2.xmin);
    float y1 = std::max(box1.ymin, box2.ymin);
    float x2 = std::min(box1.xmax, box2.xmax);
    float y2 = std::min(box1.ymax, box2.ymax);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    float overArea = w * h;
    if (overArea == 0)
        return 0.0;
    float unionArea = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - overArea;
    return overArea / unionArea;
}

std::vector<int> FaceHelper::applyNms(std::vector<BoundingBox> boxes, std::vector<float> confidences,
                                      const float nmsThresh) {
    sort(confidences.begin(), confidences.end(), [&confidences](size_t index1, size_t index2) { return confidences[index1] > confidences[index2]; });
    const int numBox = confidences.size();
    std::vector<bool> isSuppressed(numBox, false);
    for (int i = 0; i < numBox; ++i) {
        if (isSuppressed[i]) {
            continue;
        }
        for (int j = i + 1; j < numBox; ++j) {
            if (isSuppressed[j]) {
                continue;
            }

            float ovr = getIoU(boxes[i], boxes[j]);
            if (ovr > nmsThresh) {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keepInds;
    for (int i = 0; i < isSuppressed.size(); i++) {
        if (!isSuppressed[i]) {
            keepInds.emplace_back(i);
        }
    }
    return keepInds;
}

std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
FaceHelper::warpFaceByFaceLandmarks5(const VisionFrame &tempVisionFrame,
                                     const FaceLandmark &faceLandmark5,
                                     const std::vector<cv::Point2f> &warpTemplate,
                                     const cv::Size &cropSize) {
    cv::Mat affineMatrix = estimateMatrixByFaceLandmark5(faceLandmark5, warpTemplate, cropSize);
    Typing::VisionFrame cropVision;
    cv::warpAffine(tempVisionFrame, cropVision, affineMatrix, cropSize, cv::INTER_AREA, cv::BORDER_REPLICATE);
    return std::make_shared<std::tuple<VisionFrame, cv::Mat>>(cropVision, affineMatrix);
}

cv::Mat FaceHelper::estimateMatrixByFaceLandmark5(const FaceLandmark &landmark5,
                                                  const std::vector<cv::Point2f> &warpTemplate,
                                                  const cv::Size &cropSize) {
    std::vector<cv::Point2f> normedWarpTemplate = warpTemplate;
    for (auto &point : normedWarpTemplate) {
        point.x *= (float)cropSize.width;
        point.y *= (float)cropSize.height;
    }
    cv::Mat affineMatrix = cv::estimateAffinePartial2D(landmark5, normedWarpTemplate,
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

std::shared_ptr<Typing::FaceLandmark>
FaceHelper::convertFaceLandmark68To5(const FaceLandmark &faceLandmark68) {
    Typing::FaceLandmark faceLandmark5_68(5);
    faceLandmark5_68.resize(5);
    float x = 0, y = 0;
    for (int i = 36; i < 42; i++) { /// left_eye
        x += faceLandmark68[i].x;
        y += faceLandmark68[i].y;
    }
    x /= 6;
    y /= 6;
    faceLandmark5_68[0] = cv::Point2f(x, y); /// left_eye

    x = 0, y = 0;
    for (int i = 42; i < 48; i++) { /// right_eye

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

std::shared_ptr<Typing::VisionFrame>
FaceHelper::pasteBack(const cv::Mat &tempVisionFrame, const cv::Mat &cropVisionFrame,
                      const cv::Mat &cropMask, const cv::Mat &affineMatrix) {
    cv::Mat inverseMatrix;
    cv::invertAffineTransform(affineMatrix, inverseMatrix);
    cv::Mat inverseMask;
    cv::Size tempSize(tempVisionFrame.cols, tempVisionFrame.rows);
    warpAffine(cropMask, inverseMask, inverseMatrix, tempSize);
    inverseMask.setTo(0, inverseMask < 0);
    inverseMask.setTo(1, inverseMask > 1);
    cv::Mat inverseVisionFrame;
    warpAffine(cropVisionFrame, inverseVisionFrame, inverseMatrix, tempSize, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

    std::vector<cv::Mat> inverseVisionFrameBgrs(3);
    split(inverseVisionFrame, inverseVisionFrameBgrs);
    std::vector<cv::Mat> tempVisionFrameBgrs(3);
    split(tempVisionFrame, tempVisionFrameBgrs);
    for (int c = 0; c < 3; c++) {
        inverseVisionFrameBgrs[c].convertTo(inverseVisionFrameBgrs[c], CV_32FC1); ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
        tempVisionFrameBgrs[c].convertTo(tempVisionFrameBgrs[c], CV_32FC1);       ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
    }
    std::vector<cv::Mat> channelMats(3);

    channelMats[0] = inverseMask.mul(inverseVisionFrameBgrs[0]) + tempVisionFrameBgrs[0].mul(1 - inverseMask);
    channelMats[1] = inverseMask.mul(inverseVisionFrameBgrs[1]) + tempVisionFrameBgrs[1].mul(1 - inverseMask);
    channelMats[2] = inverseMask.mul(inverseVisionFrameBgrs[2]) + tempVisionFrameBgrs[2].mul(1 - inverseMask);

    cv::Mat pasteVisionFrame;
    merge(channelMats, pasteVisionFrame);
    pasteVisionFrame.convertTo(pasteVisionFrame, CV_8UC3);
    return std::make_shared<Typing::VisionFrame>(std::move(pasteVisionFrame));
}
} // namespace Ffc