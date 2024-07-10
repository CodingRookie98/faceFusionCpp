/**
 ******************************************************************************
 * @file           : face_masker.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-8
 ******************************************************************************
 */

#include "face_masker.h"

namespace Ffc {
std::shared_ptr<cv::Mat>
FaceMasker::createStaticBoxMask(const cv::Size &cropSize, const float &faceMaskBlur,
                                const Typing::Padding &faceMaskPadding) {
    int blurAmount = static_cast<int>(cropSize.width * 0.5 * faceMaskBlur);
    int blurArea = std::max(blurAmount / 2, 1);

    cv::Mat boxMask(cropSize, CV_32F, cv::Scalar(1.0f));

    int paddingTop = std::max(blurArea, static_cast<int>(cropSize.height * std::get<0>(faceMaskPadding) / 100.0));
    int paddingBottom = std::max(blurArea, static_cast<int>(cropSize.height * std::get<2>(faceMaskPadding) / 100.0));
    int paddingLeft = std::max(blurArea, static_cast<int>(cropSize.width * std::get<3>(faceMaskPadding) / 100.0));
    int paddingRight = std::max(blurArea, static_cast<int>(cropSize.width * std::get<1>(faceMaskPadding) / 100.0));

    boxMask(cv::Range(0, paddingTop), cv::Range::all()) = 0;
    boxMask(cv::Range(cropSize.height - paddingBottom, cropSize.height), cv::Range::all()) = 0;
    boxMask(cv::Range::all(), cv::Range(0, paddingLeft)) = 0;
    boxMask(cv::Range::all(), cv::Range(cropSize.width - paddingRight, cropSize.width)) = 0;

    if (blurAmount > 0) {
        cv::GaussianBlur(boxMask, boxMask, cv::Size(0, 0), blurAmount * 0.25);
    }

    return std::make_shared<cv::Mat>(std::move(boxMask));
}
} // namespace Ffc