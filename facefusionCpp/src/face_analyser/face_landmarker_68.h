/**
 ******************************************************************************
 * @file           : face_landmarker_68.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_H_

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define NOMINMAX
#endif

#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include "typing.h"
#include "face_helper.h"
#include "ort_session.h"
#include "file_system.h"
#include "downloader.h"

namespace Ffc {

class FaceLandmarker68 : public OrtSession {
public:
    explicit FaceLandmarker68(const std::shared_ptr<Ort::Env> &env,
                              const std::shared_ptr<const nlohmann::json> &modelsInfoJson);
    ~FaceLandmarker68() override = default;

    std::shared_ptr<std::tuple<Typing::FaceLandmark, Typing::Score>>
    detect(const Typing::VisionFrame &visionFrame, const Typing::BoundingBox &boundingBox);

private:
    int m_inputHeight{};
    int m_inputWidth{};
    std::tuple<std::vector<float>, cv::Mat> preProcess(const Typing::VisionFrame &visionFrame, const BoundingBox &boundingBox);
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_H_
