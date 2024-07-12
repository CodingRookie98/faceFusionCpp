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

#include <onnxruntime_cxx_api.h>
#include "opencv2/opencv.hpp"
#include "typing.h"

namespace Ffc {

class FaceMasker {
public:
    FaceMasker(const std::shared_ptr<Ort::Env> &env);
    ~FaceMasker() = default;
    static std::shared_ptr<cv::Mat> createStaticBoxMask(const cv::Size &cropSize, const float &faceMaskBlur,
                                                        const Typing::Padding &faceMaskPadding);
private:
    std::shared_ptr<Ort::Env> m_env;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_MASKER_H_
