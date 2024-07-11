/**
 ******************************************************************************
 * @file           : face_helper.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_HELPER_H_
#define FACEFUSIONCPP_SRC_FACE_HELPER_H_

#include <vector>
#include <opencv2/opencv.hpp>
#include "typing.h"

namespace Ffc {
using namespace Typing;
class FaceHelper {
public:
    static std::unordered_map<std::string, std::vector<cv::Point2f>> m_warpTemplateMap;
    FaceHelper() = default;
    ~FaceHelper() = default;

    static std::vector<int> applyNms(std::vector<BoundingBox> boxes, std::vector<float> confidences, const float nmsThresh);

    static std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
    warpFaceByFaceLandmarks5(const Typing::VisionFrame &tempVisionFrame,
                             const Typing::FaceLandmark &faceLandmark5,
                             const std::string &warpTemplate,
                             const cv::Size &cropSize);
    static std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
    warpFaceByFaceLandmarks5(const Typing::VisionFrame &tempVisionFrame,
                             const Typing::FaceLandmark &faceLandmark5,
                             const std::vector<cv::Point2f> &warpTemplate,
                             const cv::Size &cropSize);

    static cv::Mat estimateMatrixByFaceLandmark5(const Typing::FaceLandmark &landmark5,
                                                 const std::string &warpTemplate,
                                                 const cv::Size &cropSize);
    static cv::Mat estimateMatrixByFaceLandmark5(const Typing::FaceLandmark &landmark5,
                                                 const std::vector<cv::Point2f> &warpTemplate,
                                                 const cv::Size &cropSize);

    static std::shared_ptr<std::tuple<cv::Mat, cv::Mat>> warpFaceByTranslation(const cv::Mat &tempVisionFrame,
                                                                               const std::vector<float> &translation,
                                                                               const float &scale,
                                                                               const cv::Size &cropSize);
    static std::shared_ptr<Typing::FaceLandmark> convertFaceLandmark68To5(const Typing::FaceLandmark &faceLandmark68);

    static cv::Mat pasteBack(const cv::Mat &tempVisionFrame, const cv::Mat &cropVisionFrame, const cv::Mat &cropMask, const cv::Mat &affineMatrix);

private:
    static float getIoU(const BoundingBox &box1, const BoundingBox &box2);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_HELPER_H_
