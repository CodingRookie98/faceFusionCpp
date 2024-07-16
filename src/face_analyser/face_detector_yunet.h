/**
 ******************************************************************************
 * @file           : face_detector_yunet.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-16
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YUNET_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YUNET_H_

#include <nlohmann/json.hpp>
#include "ort_session.h"
#include "typing.h"
#include "filesystem"
#include "downloader.h"
#include "vision.h"
#include "globals.h"
#include "face_helper.h"

namespace Ffc {

class FaceDetectorYunet : public OrtSession {
public:
    FaceDetectorYunet(const std::shared_ptr<Ort::Env> &env,
                      const std::shared_ptr<nlohmann::json> &modelsInfoJson);
    ~FaceDetectorYunet() override = default;
    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detect(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

private:
    void preProcess(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);
    const std::shared_ptr<nlohmann::json> m_modelsInfoJson;
    int m_inputHeight{};
    int m_inputWidth{};
    float m_ratioHeight;
    float m_ratioWidth;
    std::shared_ptr<cv::FaceDetectorYN> m_faceDetectorYN;
    cv::Mat m_inputVisionFrame;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YUNET_H_
