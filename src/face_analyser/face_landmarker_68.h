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

#include <opencv2/imgproc.hpp>
#include "analyser_base.h"
#include "typing.h"
#include "face_analyser_session.h"
#include "face_helper.h"

namespace Ffc {

class FaceLandmarker68 : public AnalyserBase, public FaceAnalyserSession {
public:
    FaceLandmarker68(const std::shared_ptr<Ort::Env> &env);
    ~FaceLandmarker68() = default;

    std::shared_ptr<std::tuple<Typing::FaceLandmark, Typing::Score>>
    detect(const Typing::VisionFrame &visionFrame, const Typing::BoundingBox &boundingBox);

private:
    std::vector<float> m_inputImage;
    int m_inputHeight{};
    int m_inputWidth{};
    cv::Mat m_invAffineMatrix;
    void preProcess(const Typing::VisionFrame &visionFrame, const BoundingBox &boundingBox);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_FACE_LANDMARKER_68_H_
