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

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "typing.h"
#include "config.h"
#include "ort_session.h"
#include "logger.h"
#include "face_detector_yolo.h"
#include "face_landmarker_68.h"
#include "face_landmarker_68_5.h"
#include "face_recognizer_arc.h"
#include "face_detector_gender_age.h"
#include "face_detector_scrfd.h"
#include "face_detector_retina.h"
#include "face_detector_yunet.h"
#include "face_store.h"

namespace Ffc {

class FaceAnalyser {
public:
    enum Method {
        DetectWithYoloFace,
        DetectWithScrfd,
        DetectWithRetina,
        DetectWithYunet,
        DetectLandmark68,
        DetectLandmark68_5,
        RecognizeWithArcFace,
        DetectorGenderAge
    };

    explicit FaceAnalyser(const std::shared_ptr<Ort::Env> &env,
                          const std::shared_ptr<const nlohmann::json> &modelsInfoJson,
                          const std::shared_ptr<const Config> &config);
    ~FaceAnalyser() = default;

    std::shared_ptr<Typing::Face> getAverageFace(const std::vector<Typing::VisionFrame> &visionFrames,
                                                 const int &position = 0);

    std::shared_ptr<Typing::Faces> getManyFaces(const Typing::VisionFrame &visionFrame);

    std::shared_ptr<Typing::Face> getOneFace(const Typing::VisionFrame &visionFrame,
                                             const int &position = 0);
    
    Typing::Faces findSimilarFaces(const Typing::Faces &referenceFaces,
                                   const Typing::VisionFrame &targetVisionFrame,
                                   const float &faceDistance);
    
    bool compareFace(const Typing::Face &face,
                     const Typing::Face &referenceFace,
                     const float &faceDistance);
    
    static float calculateFaceDistance(const Typing::Face &face1,
                                const Typing::Face &face2);

    bool preCheck();

private:
    std::shared_ptr<Ort::Env> m_env;
    const std::shared_ptr<const nlohmann::json> m_modelsInfoJson;
    std::unordered_map<Method, std::shared_ptr<OrtSession>> m_analyserMap;
    std::mutex m_mutex;
    const std::shared_ptr<const Config> m_config;
    std::shared_ptr<Logger> m_logger = Logger::getInstance();
    std::shared_ptr<FaceStore> m_faceStore = FaceStore::getInstance();
    
    std::shared_ptr<OrtSession> getAnalyser(const Method &method);
    
    std::shared_ptr<Typing::Faces> createFaces(const Typing::VisionFrame &visionFrame,
                                               const std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                                                                                std::vector<Typing::FaceLandmark>,
                                                                                std::vector<Typing::Score>>> &input);
    std::shared_ptr<Typing::FaceLandmark> expandFaceLandmark68From5(const Typing::FaceLandmark &inputLandmark5);

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detectWithYoloFace(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detectWithScrfd(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detectWithRetina(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

    std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                               std::vector<Typing::FaceLandmark>,
                               std::vector<Typing::Score>>>
    detectWithYunet(const Typing::VisionFrame &visionFrame, const cv::Size &faceDetectorSize);

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

    static std::shared_ptr<Typing::Faces>
    sortByOrder(std::shared_ptr<Typing::Faces> faces,
                const Typing::EnumFaceSelectorOrder &order);

    static std::shared_ptr<Typing::Faces>
    filterByAge(std::shared_ptr<Typing::Faces> faces, const Typing::EnumFaceSelectorAge &age);

    static std::shared_ptr<Typing::Faces>
    filterByGender(std::shared_ptr<Typing::Faces> faces, const Typing::EnumFaceSelectorGender &gender);
};
} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FACE_ANALYSER_H_
