/**
 ******************************************************************************
 * @file           : face_analyser.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-3
 ******************************************************************************
 */

#include "face_analyser.h"

namespace Ffc {
FaceAnalyser::FaceAnalyser(const std::shared_ptr<Ort::Env> &env, const std::shared_ptr<nlohmann::json> &modelsInfoJson) {
    this->m_env = env;
    m_modelsInfoJson = modelsInfoJson;
}

void FaceAnalyser::createAnalyser(const FaceAnalyser::Method &method) {
    if (m_analyserMap.contains(method)) {
        return;
    }

    std::shared_ptr<OrtSession> analyserPtr = nullptr;
    switch (method) {
    case DetectWithYoloFace:
        analyserPtr = std::make_shared<FaceDetectorYolo>(m_env, m_modelsInfoJson);
        break;
    case DetectWithScrfd:
        analyserPtr = std::make_shared<FaceDetectorScrfd>(m_env, m_modelsInfoJson);
        break;
    case DetectWithRetina:
        analyserPtr = std::make_shared<FaceDetectorRetina>(m_env, m_modelsInfoJson);
        break;
    case DetectWithYunet:
        analyserPtr = std::make_shared<FaceDetectorYunet>(m_env, m_modelsInfoJson);
        break;
    case DetectLandmark68:
        analyserPtr = std::make_shared<FaceLandmarker68>(m_env, m_modelsInfoJson);
        break;
    case DetectLandmark68_5:
        analyserPtr = std::make_shared<FaceLandmarker68_5>(m_env, m_modelsInfoJson);
        break;
    case RecognizeWithArcfaceW600kR50:
        analyserPtr = std::make_shared<FaceRecognizerArcW600kR50>(m_env, m_modelsInfoJson);
        break;
    case DetectorGenderAge:
        analyserPtr = std::make_shared<FaceDetectorGenderAge>(m_env);
        break;
    default: break;
    }

    if (analyserPtr == nullptr) {
        throw std::runtime_error("FaceAnalyser::createAnalyser: method not supported or std::make_shared failed");
    } else {
        m_analyserMap.insert(std::make_pair(method, std::move(analyserPtr)));
    }
}

std::shared_ptr<Typing::Face>
FaceAnalyser::getAverageFace(const std::vector<Typing::VisionFrame> &visionFrames, const int &position) {
    Typing::Face averageFace;
    Typing::Faces faces{};

    for (const auto &visionFrame : visionFrames) {
        auto face = this->getOneFace(visionFrame, position);
        if (face != nullptr && !face->isEmpty()) {
            faces.push_back(std::move(*face));
        }
    }

    if (!faces.empty()) {
        Typing::Face firstFace = faces.front();
        averageFace.boundingBox = firstFace.boundingBox;
        averageFace.faceLandmark5 = firstFace.faceLandmark5;
        averageFace.faceLandMark5_68 = firstFace.faceLandMark5_68;
        averageFace.faceLandmark68 = firstFace.faceLandmark68;
        averageFace.faceLandmark68_5 = firstFace.faceLandmark68_5;
        averageFace.detectorScore = firstFace.detectorScore;
        averageFace.landmarkerScore = firstFace.landmarkerScore;

        if (faces.size() > 1) {
            Typing::Embedding averageEmbedding(faces.front().embedding.size());
            Typing::Embedding averageNormedEmbedding(faces.front().normedEmbedding.size());
            for (auto &face : faces) {
                for (int j = 0; j < face.embedding.size(); ++j) {
                    averageEmbedding.at(j) += face.embedding.at(j);
                    averageNormedEmbedding.at(j) += face.normedEmbedding.at(j);
                }
            }
            for (int j = 0; j < averageEmbedding.size(); ++j) {
                averageEmbedding.at(j) /= (float)faces.size();
                averageNormedEmbedding.at(j) /= (float)faces.size();
            }
            averageFace.embedding = averageEmbedding;
            averageFace.normedEmbedding = averageNormedEmbedding;
        } else {
            averageFace.embedding = firstFace.embedding;
            averageFace.normedEmbedding = firstFace.normedEmbedding;
        }
    }

    return std::make_shared<Typing::Face>(std::move(averageFace));
}

std::shared_ptr<Typing::Face> FaceAnalyser::getOneFace(const Typing::VisionFrame &visionFrame, const int &position) {
    std::shared_ptr<Typing::Faces> manyFaces = this->getManyFaces(visionFrame);
    if (!manyFaces->empty()) {
        if (position < 0 || position >= manyFaces->size()) {
            if (!manyFaces->empty()) {
                return std::make_shared<Typing::Face>(std::move(manyFaces->back()));
            } else {
                throw std::runtime_error("FaceAnalyser::getOneFace: position out of range");
            }
        } else {
            return std::make_shared<Typing::Face>(std::move(manyFaces->at(position)));
        }
    }
    return {};
}

std::shared_ptr<Typing::Faces> FaceAnalyser::getManyFaces(const Typing::VisionFrame &visionFrame) {
    std::vector<Typing::BoundingBox> resultBoundingBoxes, boundingBoxes;
    std::vector<Typing::FaceLandmark> resultLandmarks5, landmarks5;
    std::vector<Typing::Score> resultScores, scores;
    std::shared_ptr<Typing::Faces> resultFaces = std::make_shared<Typing::Faces>();
    if (Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Many)
        || Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Yoloface)) {
        auto result = this->detectWithYoloFace(visionFrame, Globals::faceDetectorSize);
        std::tie(boundingBoxes, landmarks5, scores) = *result;
        resultBoundingBoxes.insert(resultBoundingBoxes.end(), boundingBoxes.begin(), boundingBoxes.end());
        resultLandmarks5.insert(resultLandmarks5.end(), landmarks5.begin(), landmarks5.end());
        resultScores.insert(resultScores.end(), scores.begin(), scores.end());
    }
    if (Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Many)
        || Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Scrfd)) {
        auto result = this->detectWithScrfd(visionFrame, Globals::faceDetectorSize);
        std::tie(boundingBoxes, landmarks5, scores) = *result;
        resultBoundingBoxes.insert(resultBoundingBoxes.end(), boundingBoxes.begin(), boundingBoxes.end());
        resultLandmarks5.insert(resultLandmarks5.end(), landmarks5.begin(), landmarks5.end());
        resultScores.insert(resultScores.end(), scores.begin(), scores.end());
    }
    if (Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Many)
        || Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Retina)) {
        auto result = this->detectWithRetina(visionFrame, Globals::faceDetectorSize);
        std::tie(boundingBoxes, landmarks5, scores) = *result;
        resultBoundingBoxes.insert(resultBoundingBoxes.end(), boundingBoxes.begin(), boundingBoxes.end());
        resultLandmarks5.insert(resultLandmarks5.end(), landmarks5.begin(), landmarks5.end());
        resultScores.insert(resultScores.end(), scores.begin(), scores.end());
    }
    if (Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Many)
        || Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Yunet)) {
        auto result = this->detectWithYunet(visionFrame, Globals::faceDetectorSize);
        std::tie(boundingBoxes, landmarks5, scores) = *result;
        resultBoundingBoxes.insert(resultBoundingBoxes.end(), boundingBoxes.begin(), boundingBoxes.end());
        resultLandmarks5.insert(resultLandmarks5.end(), landmarks5.begin(), landmarks5.end());
        resultScores.insert(resultScores.end(), scores.begin(), scores.end());
    }

    if (!resultBoundingBoxes.empty() && !resultLandmarks5.empty() && !resultScores.empty()) {
        resultFaces = createFaces(visionFrame, std::make_shared<std::tuple<std::vector<Typing::BoundingBox>,
                                                                           std::vector<Typing::FaceLandmark>,
                                                                           std::vector<Typing::Score>>>(
                                                   std::make_tuple(resultBoundingBoxes, resultLandmarks5, resultScores)));
    }

    // Todo 对faces排序以及按照年龄和性别筛选

    return resultFaces;
}

std::shared_ptr<Typing::FaceLandmark>
FaceAnalyser::expandFaceLandmark68From5(const FaceLandmark &inputLandmark5) {
    if (!m_analyserMap.contains(DetectLandmark68_5)) {
        createAnalyser(DetectLandmark68_5);
    }

    auto detectorLandmark68_5 = std::dynamic_pointer_cast<FaceLandmarker68_5>(m_analyserMap.at(DetectLandmark68_5));
    return detectorLandmark68_5->detect(inputLandmark5);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceAnalyser::detectWithYoloFace(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    if (!m_analyserMap.contains(DetectWithYoloFace)) {
        createAnalyser(DetectWithYoloFace);
    }

    auto detectorYolo = std::dynamic_pointer_cast<FaceDetectorYolo>(m_analyserMap.at(DetectWithYoloFace));
    return detectorYolo->detect(visionFrame, faceDetectorSize);
}

std::shared_ptr<Typing::Faces>
FaceAnalyser::createFaces(const Typing::VisionFrame &visionFrame,
                          const std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                                                           std::vector<Typing::FaceLandmark>,
                                                           std::vector<Typing::Score>>> &input) {
    std::shared_ptr<Typing::Faces> resultFaces = std::make_shared<Typing::Faces>();

    if (Ffc::Globals::faceDetectorScore <= 0) {
        return resultFaces;
    }

    auto boundingBoxes = new std::vector<Typing::BoundingBox>(std::get<0>(*input));
    auto faceLandmarks = new std::vector<Typing::FaceLandmark>(std::get<1>(*input));
    auto scores = new std::vector<Typing::Score>(std::get<2>(*input));

    float iouThreshold = Globals::faceDetectorModelSet.contains(Typing::EnumFaceDetectModel::FD_Many) ? 0.1 : 0.4;
    auto keepIndices = Ffc::FaceHelper::applyNms(*boundingBoxes, *scores, iouThreshold);

    for (const auto &index : keepIndices) {
        Typing::Face tempFace;
        tempFace.boundingBox = boundingBoxes->at(index);
        tempFace.faceLandMark5_68 = faceLandmarks->at(index);
        tempFace.faceLandmark5 = faceLandmarks->at(index);
        tempFace.faceLandmark68_5 = *expandFaceLandmark68From5(tempFace.faceLandMark5_68);
        tempFace.faceLandmark68 = tempFace.faceLandmark68_5;
        tempFace.detectorScore = scores->at(index);
        tempFace.landmarkerScore = 0.0;
        if (Ffc::Globals::faceLandmarkerScore > 0) {
            auto faceLandmark68AndScore = this->detectLandmark68(visionFrame, tempFace.boundingBox);
            if (faceLandmark68AndScore != nullptr) {
                tempFace.faceLandmark68 = std::get<0>(*faceLandmark68AndScore);
                tempFace.landmarkerScore = std::get<1>(*faceLandmark68AndScore);
                if (std::get<1>(*faceLandmark68AndScore) > Ffc::Globals::faceLandmarkerScore) {
                    tempFace.faceLandMark5_68 = *(FaceHelper::convertFaceLandmark68To5(tempFace.faceLandmark68));
                }
            }
        }

        auto embeddingAndNormedEmbedding = this->calculateEmbedding(visionFrame, tempFace.faceLandMark5_68);
        tempFace.embedding = std::get<0>(*embeddingAndNormedEmbedding);
        tempFace.normedEmbedding = std::get<1>(*embeddingAndNormedEmbedding);

        auto ganderAge = this->detectGenderAge(visionFrame, tempFace.boundingBox);
        tempFace.gender = std::get<0>(*ganderAge);
        tempFace.age = std::get<1>(*ganderAge);

        resultFaces->emplace_back(tempFace);
    }

    delete boundingBoxes;
    delete faceLandmarks;
    delete scores;

    return resultFaces;
}

std::shared_ptr<std::tuple<Typing::FaceLandmark,
                           Typing::Score>>
FaceAnalyser::detectLandmark68(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    if (!m_analyserMap.contains(DetectLandmark68)) {
        createAnalyser(DetectLandmark68);
    }

    auto detectorLandmark68 = std::dynamic_pointer_cast<FaceLandmarker68>(m_analyserMap.at(DetectLandmark68));
    return detectorLandmark68->detect(visionFrame, boundingBox);
}

std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>> FaceAnalyser::calculateEmbedding(const VisionFrame &visionFrame, const FaceLandmark &faceLandmark5_68) {
    if (!m_analyserMap.contains(RecognizeWithArcfaceW600kR50)) {
        createAnalyser(RecognizeWithArcfaceW600kR50);
    }

    auto recognizerArc = std::dynamic_pointer_cast<FaceRecognizerArcW600kR50>(m_analyserMap.at(RecognizeWithArcfaceW600kR50));
    return recognizerArc->recognize(visionFrame, faceLandmark5_68);
}

std::shared_ptr<std::tuple<int, int>>
FaceAnalyser::detectGenderAge(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    if (!m_analyserMap.contains(DetectorGenderAge)) {
        createAnalyser(DetectorGenderAge);
    }

    auto detectorGenderAge = std::dynamic_pointer_cast<FaceDetectorGenderAge>(m_analyserMap.at(DetectorGenderAge));
    return detectorGenderAge->detect(visionFrame, boundingBox);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceAnalyser::detectWithScrfd(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    if (!m_analyserMap.contains(DetectWithScrfd)) {
        createAnalyser(DetectWithScrfd);
    }
    auto detectorScrfd = std::dynamic_pointer_cast<FaceDetectorScrfd>(m_analyserMap.at(DetectWithScrfd));
    return detectorScrfd->detect(visionFrame, faceDetectorSize);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceAnalyser::detectWithRetina(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    if (!m_analyserMap.contains(DetectWithRetina)) {
        createAnalyser(DetectWithRetina);
    }
    auto detectorRetina = std::dynamic_pointer_cast<FaceDetectorRetina>(m_analyserMap.at(DetectWithRetina));
    return detectorRetina->detect(visionFrame, faceDetectorSize);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceAnalyser::detectWithYunet(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    if (!m_analyserMap.contains(DetectWithYunet)) {
        createAnalyser(DetectWithYunet);
    }
    auto detectorYunet = std::dynamic_pointer_cast<FaceDetectorYunet>(m_analyserMap.at(DetectWithYunet));
    return detectorYunet->detect(visionFrame, faceDetectorSize);
}
} // namespace Ffc