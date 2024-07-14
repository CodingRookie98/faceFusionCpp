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

    std::shared_ptr<AnalyserBase> analyserPtr = nullptr;
    switch (method) {
    case DetectWithYoloFace:
        analyserPtr = std::make_shared<FaceDetectorYolo>(m_env);
        break;
    case DetectLandmark68:
        analyserPtr = std::make_shared<FaceLandmarker68>(m_env);
        break;
    case DetectLandmark68_5:
        analyserPtr = std::make_shared<FaceLandmarker68_5>(m_env);
        break;
    case RecognizeWithArcfaceW600kR50:
        analyserPtr = std::make_shared<FaceRecognizerArcW600kR50>(m_env);
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
        Typing::Face face = this->getOneFace(visionFrame, position);
        if (!face.isEmpty()) {
            faces.push_back(face);
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

Typing::Face FaceAnalyser::getOneFace(const Typing::VisionFrame &visionFrame, const int &position) {
    std::shared_ptr<Typing::Faces> manyFaces = this->getManyFaces(visionFrame);
    if (!manyFaces->empty()) {
        if (position < 0 || position >= manyFaces->size()) {
            if (!manyFaces->empty()) {
                return manyFaces->back();
            } else {
                throw std::runtime_error("FaceAnalyser::getOneFace: position out of range");
            }
        } else {
            return manyFaces->at(position);
        }
    }
    return Typing::Face{};
}

std::shared_ptr<Typing::Faces> FaceAnalyser::getManyFaces(const Typing::VisionFrame &visionFrame) {
    auto result = this->detectWithYoloFace(visionFrame, Globals::faceDetectorSize);
    auto faces = this->createFaces(visionFrame, result);
    
    //Todo 对faces排序以及按照年龄和性别筛选

    return faces;
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

    float iouThreshold = Globals::faceDetectorModel == Typing::EnumFaceDetectModel::FD_Many ? 0.1 : 0.4;
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
    float boundingBoxW = boundingBox.xmax - boundingBox.xmin;
    float boundingBoxH = boundingBox.ymax - boundingBox.ymin;
    float maxSide = std::max(boundingBoxW, boundingBoxH);
    float scale = (float)64 / (float)maxSide;
    std::vector<float> translation;
    translation.emplace_back(48 - scale * (boundingBox.xmin + boundingBox.xmax) * 0.5);
    translation.emplace_back(48 - scale * (boundingBox.ymin + boundingBox.ymax) * 0.5);
    auto cropVisionAndAffineMat = FaceHelper::warpFaceByTranslation(visionFrame, translation,
                                                                    scale, cv::Size(96, 96));

    std::string modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("gender_age").at("path").get<std::string>();
    auto session = std::make_shared<OrtSession>(m_env);
    session->createSession(modelPath);
    int inputHeight = session->m_inputNodeDims[0][2];
    int inputWidth = session->m_inputNodeDims[0][3];

    std::vector<cv::Mat> bgrChannels(3);
    split(std::get<0>(*cropVisionAndAffineMat), bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1);
    }
    const int imageArea = inputHeight * inputWidth;
    std::vector<float> inputImageData(3 * imageArea);
    inputImageData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize); /// rgb顺序
    memcpy(inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);

    std::vector<int64_t> inputTensorShape = {1, 3, 96, 96};
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(session->m_memoryInfo,
                                                              inputImageData.data(), inputImageData.size(),
                                                              inputTensorShape.data(), inputTensorShape.size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensor = session->m_session->Run(runOptions, session->m_inputNames.data(),
                                                                   inputTensors.data(), inputTensors.size(),
                                                                   session->m_outputNames.data(), session->m_outputNames.size());
    const float *pdta = outputTensor[0].GetTensorMutableData<float>();
    std::shared_ptr<std::tuple<int, int>> result;
    if (*(pdta) > *(pdta + 1)) {
        result = std::make_shared<std::tuple<int, int>>(0, std::round(*(pdta + 2) * 100));
    } else {
        result = std::make_shared<std::tuple<int, int>>(1, std::round(*(pdta + 2) * 100));
    }
    return result;
}
} // namespace Ffc