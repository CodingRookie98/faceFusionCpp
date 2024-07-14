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

#include <unordered_set>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include "opencv2/opencv.hpp"
#include "typing.h"
#include "ort_session.h"
#include "globals.h"

namespace Ffc {
class FaceMasker : public OrtSession {
public:
    FaceMasker(const std::shared_ptr<Ort::Env> &env,
               const std::shared_ptr<nlohmann::json> &modelsInfoJson);
    ~FaceMasker() = default;
    static std::shared_ptr<cv::Mat> createStaticBoxMask(const cv::Size &cropSize, const float &faceMaskBlur,
                                                        const Typing::Padding &faceMaskPadding);
    std::shared_ptr<cv::Mat> createOcclusionMask(const Typing::VisionFrame &cropVisionFrame);
    std::shared_ptr<cv::Mat> createRegionMask(const Typing::VisionFrame &cropVisionFrame,
                                                     const std::unordered_set<Typing::EnumFaceMaskRegion> &regions = Globals::faceMaskRegionsSet);

    static std::shared_ptr<cv::Mat> getBestMask(const std::vector<cv::Mat> &masks);

private:
    std::shared_ptr<nlohmann::json> m_modelsJson = nullptr;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_MASKER_H_
