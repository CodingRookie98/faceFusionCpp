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

#include "typing.h"
#include "analyser_base.h"
#include "face_analyser_session.h"
#include "face_helper.h"

namespace Ffc {

class FaceRecognizerArcW600kR50 : public AnalyserBase, public FaceAnalyserSession{
public:
    FaceRecognizerArcW600kR50(const std::shared_ptr<Ort::Env> &env);
    ~FaceRecognizerArcW600kR50() = default;
    
    std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
    recognize(const Typing::VisionFrame &visionFrame, const Typing::FaceLandmark &faceLandmark5);
    
private:
    void preProcess(const Typing::VisionFrame &visionFrame,
                    const Typing::FaceLandmark &faceLandmark5_68);
    std::vector<float> m_inputData;
    int m_inputWidth{};
    int m_inputHeight{};
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_RECOGNIZER_ARC_H_
