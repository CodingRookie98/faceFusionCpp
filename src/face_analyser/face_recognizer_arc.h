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

class FaceRecognizerArc : public OrtSession {
public:
    enum ArcType {
        W600k_R50,
        Simswap,
    };
    
    explicit FaceRecognizerArc(const std::shared_ptr<Ort::Env> &env,
                               const std::shared_ptr<const nlohmann::json> &modelsInfoJson,
                               const ArcType &arcType);
    ~FaceRecognizerArc() override = default;

    std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
    recognize(const Typing::VisionFrame &visionFrame, const Typing::FaceLandmark &faceLandmark5);
    ArcType getArcType() const;

private:
    void preProcess(const Typing::VisionFrame &visionFrame,
                    const Typing::FaceLandmark &faceLandmark5_68);
    std::vector<float> m_inputData;
    int m_inputWidth{};
    int m_inputHeight{};
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
    ArcType m_arcType;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_RECOGNIZER_ARC_H_
