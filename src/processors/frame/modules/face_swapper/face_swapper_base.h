/**
 ******************************************************************************
 * @file           : face_swapper_base.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-9
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_BASE_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_BASE_H_

#include "typing.h"
#include "face_helper.h"
#include "globals.h"
#include "face_masker.h"

namespace Ffc {

class FaceSwapperBase {
public:
    FaceSwapperBase() = default;
    virtual ~FaceSwapperBase() = default;

protected:
    static std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
    getCropVisionFrameAndAffineMat(const Typing::VisionFrame &visionFrame,
                                   const Typing::FaceLandmark &faceLandmark,
                                   const std::string &modelTemplate, const cv::Size &size);
    // 颜色空间转换(BGR to RGB), 标准化，归一化
    static std::shared_ptr<Typing::VisionFrame>
    prepareCropVisionFrame(const Typing::VisionFrame &visionFrame,
                           const std::vector<float> &mean,
                           const std::vector<float> &standDeviation);

    static std::shared_ptr<std::list<cv::Mat>>
    getCropMaskList(const Typing::VisionFrame &visionFrame, const cv::Size &cropSize,
                    const float &faceMaskBlur, const Typing::Padding &faceMaskPadding);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_BASE_H_
