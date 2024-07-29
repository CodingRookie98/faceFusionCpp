/**
 ******************************************************************************
 * @file           : face_detector_gender_age.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-15
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_GENDER_AGE_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_GENDER_AGE_H_

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include "typing.h"
#include "ort_session.h"
#include "face_helper.h"
#include "vision.h"
#include "file_system.h"
#include "downloader.h"

namespace Ffc {

class FaceDetectorGenderAge : public OrtSession {
public:
    FaceDetectorGenderAge(const std::shared_ptr<Ort::Env> &env,
                          const std::shared_ptr<const nlohmann::json> &modelsInfoJson);
    ~FaceDetectorGenderAge() override = default;

    std::shared_ptr<std::tuple<int, int>>
    detect(const Typing::VisionFrame &visionFrame,
                    const Typing::BoundingBox &boundingBox);

private:
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
    int m_inputHeight;
    int m_inputWidth;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_DETECTOR_GENDER_AGE_H_
