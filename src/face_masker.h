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
#include "file_system.h"
#include "downloader.h"

namespace Ffc {
class FaceMasker {
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
    enum Method {
        Occlusion,
        Region
    };
    std::shared_ptr<Ort::Env> m_env = nullptr;
    const std::shared_ptr<nlohmann::json> m_modelsInfoJson;
    std::shared_ptr<std::unordered_map<Method, std::shared_ptr<OrtSession>>> m_maskerMap;
};
} // namespace Ffc
#endif // FACEFUSIONCPP_SRC_FACE_MASKER_H_
