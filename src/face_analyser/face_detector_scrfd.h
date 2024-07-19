/**
 ******************************************************************************
 * @file           : face_detector_scrfd.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-16
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_SCRFD_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_SCRFD_H_

#include <nlohmann/json.hpp>
#include "ort_session.h"
#include "typing.h"
#include "filesystem"
#include "downloader.h"
#include "vision.h"
#include "face_helper.h"

namespace Ffc {

class FaceDetectorScrfd : public OrtSession {
public:
    FaceDetectorScrfd(const std::shared_ptr<Ort::Env> &env,
                      const std::shared_ptr<const nlohmann::json> &modelsInfoJson);
    ~FaceDetectorScrfd() override = default;

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detect(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize,
           const float &detectorScore = 0.5);

private:
    void preProcess(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
    std::vector<float> m_inputData;
    int m_inputHeight{};
    int m_inputWidth{};
    float m_ratioHeight;
    float m_ratioWidth;
    const std::vector<int> m_featureStrides = {8, 16, 32};
    const int m_featureMapChannel = 3;
    const int m_anchorTotal = 2;
};
} // namespace Ffc
#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_SCRFD_H_
