/**
 ******************************************************************************
 * @file           : face_yolov_8.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YOLO_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YOLO_H_

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "typing.h"
#include "ort_session.h"
#include "face_helper.h"
#include "vision.h"
#include "file_system.h"
#include "downloader.h"
#include "config.h"

namespace Ffc {

class FaceDetectorYolo : public OrtSession {
public:
    explicit FaceDetectorYolo(const std::shared_ptr<Ort::Env> &env,
                              const std::shared_ptr<const nlohmann::json> &modelsInfoJson);
    ~FaceDetectorYolo() override = default;

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detect(const Typing::VisionFrame &visionFrame,const cv::Size &faceDetectorSize,
           const float &scoreThreshold = 0.5);

private:
    std::tuple<std::vector<float>, float, float> preProcess(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);
    int m_inputHeight{};
    int m_inputWidth{};
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_YOLO_H_
