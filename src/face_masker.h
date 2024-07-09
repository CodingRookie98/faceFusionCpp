/**
 ******************************************************************************
 * @file           : face_masker.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-8
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_MASKER_H_
#define FACEFUSIONCPP_SRC_FACE_MASKER_H_

#include "opencv2/opencv.hpp"
#include "typing.h"

namespace Ffc {

class FaceMasker {
public:
    FaceMasker() = default;
    ~FaceMasker() = default;
    static cv::Mat createStaticBoxMask(const cv::Size &cropSize, const float faceMaskBlur,
                                       const Typing::Padding &faceMaskPadding);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_MASKER_H_
