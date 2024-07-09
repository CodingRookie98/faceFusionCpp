/**
 ******************************************************************************
 * @file           : face_landmarker_68_5.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_5_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_5_H_

#include "analyser_base.h"
#include "face_analyser_session.h"
#include "face_helper.h"

namespace Ffc {

class FaceLandmarker68_5 : public Ffc::AnalyserBase, public Ffc::FaceAnalyserSession {
public:
    FaceLandmarker68_5(const std::shared_ptr<Ort::Env> &env);
    ~FaceLandmarker68_5() override = default;

    std::shared_ptr<Typing::FaceLandmark > detect(const Typing::FaceLandmark &faceLandmark5);

private:
    void preProcess(const Typing::FaceLandmark &faceLandmark5);
    std::vector<float> m_inputTensorData;
    int m_inputHeight{};
    int m_inputWidth{};
    cv::Mat m_affineMatrix;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_5_H_
