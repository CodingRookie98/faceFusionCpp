/**
 ******************************************************************************
 * @file           : face_analyser.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-3
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FACE_ANALYSER_H_
#define FACEFUSIONCPP_SRC_FACE_ANALYSER_H_

#include <opencv2/imgproc.hpp>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "typing.h"
#include "face_detector_yolo.h"
#include "face_landmarker_68.h"
#include "face_landmarker_68_5.h"
#include "face_recognizer_arc.h"
#include "globals.h"
#include "ort_session.h"
#include "face_detector_gender_age.h"

namespace Ffc {

class FaceAnalyser {
public:
    enum Method {
        DetectWithYoloFace,
        DetectLandmark68,
        DetectLandmark68_5,
        RecognizeWithArcfaceW600kR50,
        DetectorGenderAge
    };

    explicit FaceAnalyser(const std::shared_ptr<Ort::Env> &env, const std::shared_ptr<nlohmann::json> &modelsInfoJson);
    ~FaceAnalyser() = default;

    std::shared_ptr<Typing::Face> getAverageFace(const std::vector<Typing::VisionFrame> &visionFrames, const int &position = 0);

    std::shared_ptr<Typing::Faces> getManyFaces(const Typing::VisionFrame &visionFrame);

    std::shared_ptr<Typing::Face> getOneFace(const Typing::VisionFrame &visionFrame, const int &position = 0);

private:
    std::shared_ptr<Ort::Env> m_env;

    std::shared_ptr<nlohmann::json> m_modelsInfoJson;
    std::unordered_map<Method, std::shared_ptr<OrtSession>> m_analyserMap;
    void createAnalyser(const Method &method);
    std::shared_ptr<Typing::Faces> createFaces(const Typing::VisionFrame &visionFrame,
                                               const std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                                                                                std::vector<Typing::FaceLandmark>,
                                                                                std::vector<Typing::Score>>> &input);
    std::shared_ptr<Typing::FaceLandmark> expandFaceLandmark68From5(const Typing::FaceLandmark &inputLandmark5);

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detectWithYoloFace(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

    std::shared_ptr<std::tuple<Typing::FaceLandmark,
                               Typing::Score>>
    detectLandmark68(const Typing::VisionFrame &visionFrame,
                     const Typing::BoundingBox &boundingBox);
    std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
    calculateEmbedding(const Typing::VisionFrame &visionFrame,
                       const Typing::FaceLandmark &faceLandmark5_68);
    std::shared_ptr<std::tuple<int, int>>
    detectGenderAge(const Typing::VisionFrame &visionFrame,
                    const Typing::BoundingBox &boundingBox);
};
} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_H_
