/**
 ******************************************************************************
 * @file           : face_recognizer_arc.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-8
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_RECOGNIZER_ARC_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_RECOGNIZER_ARC_H_

#include <nlohmann/json.hpp>
#include "typing.h"
#include "face_helper.h"
#include "ort_session.h"
#include "file_system.h"
#include "downloader.h"

namespace Ffc {

class FaceRecognizerArcW600kR50 : public OrtSession {
public:
    explicit FaceRecognizerArcW600kR50(const std::shared_ptr<Ort::Env> &env,
                                       const std::shared_ptr<nlohmann::json> &modelsInfoJson);
    ~FaceRecognizerArcW600kR50() override = default;

    std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
    recognize(const Typing::VisionFrame &visionFrame, const Typing::FaceLandmark &faceLandmark5);

private:
    void preProcess(const Typing::VisionFrame &visionFrame,
                    const Typing::FaceLandmark &faceLandmark5_68);
    std::vector<float> m_inputData;
    int m_inputWidth{};
    int m_inputHeight{};
    const std::shared_ptr<nlohmann::json> m_modelsInfoJson;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_RECOGNIZER_ARC_H_
