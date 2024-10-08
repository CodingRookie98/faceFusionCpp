/**
 ******************************************************************************
 * @file           : face_recognizer_arc.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-8
 ******************************************************************************
 */

#include "face_recognizer_arc.h"

namespace Ffc {
FaceRecognizerArc::FaceRecognizerArc(const std::shared_ptr<Ort::Env> &env,
                                     const std::shared_ptr<const nlohmann::json> &modelsInfoJson,
                                     const ArcType &arcType) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    m_arcType = arcType;
    std::string modelPath;
    switch (m_arcType) {
    case W600k_R50:
        modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("face_recognizer_arcface_blendswap").at("path");
        break;
    case Simswap:
        modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("face_recognizer_arcface_simswap").at("path");
        break;
    }
    // 如果 modelPath不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceAnalyserModels").at("face_recognizer_arcface_uniface").at("url"),
                                                    "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    this->createSession(modelPath);
}

FaceRecognizerArc::ArcType FaceRecognizerArc::getArcType() const {
    return m_arcType;
}

std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
FaceRecognizerArc::recognize(const Typing::VisionFrame &visionFrame,
                             const Typing::FaceLandmark &faceLandmark5) {
    std::vector<float> inputData = this->preProcess(visionFrame, faceLandmark5);
    std::vector<int64_t> inputImgShape = {1, 3, this->m_inputHeight, this->m_inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(this->m_memoryInfo, inputData.data(),
                                                             inputData.size(), inputImgShape.data(),
                                                             inputImgShape.size());

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = this->m_session->Run(runOptions, this->m_inputNames.data(), &inputTensor, 1, this->m_outputNames.data(), m_outputNames.size());

    float *pdata = ortOutputs[0].GetTensorMutableData<float>(); /// 形状是(1, 512)
    const int lenFeature = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];

    Typing::Embedding embedding(lenFeature), normedEmbedding(lenFeature);

    memcpy(embedding.data(), pdata, lenFeature * sizeof(float));

    double norm = cv::norm(embedding, cv::NORM_L2);
    for (int i = 0; i < lenFeature; i++) {
        normedEmbedding.at(i) = embedding.at(i) / (float)norm;
    }

    return std::make_shared<std::tuple<Typing::Embedding, Typing::Embedding>>(std::move(embedding),
                                                                              std::move(normedEmbedding));
}

std::vector<float> FaceRecognizerArc::preProcess(const Typing::VisionFrame &visionFrame,
                                                 const Typing::FaceLandmark &faceLandmark5_68) {
    m_inputWidth = (int)m_inputNodeDims[0][2];
    m_inputHeight = (int)m_inputNodeDims[0][3];

    std::vector<float> fVec = m_modelsInfoJson->at("faceHelper").at("warpTemplate").at("arcface_112_v2").get<std::vector<float>>();
    std::vector<cv::Point2f> warpTemplate;
    for (int i = 0; i < fVec.size(); i += 2) {
        warpTemplate.emplace_back(fVec.at(i), fVec.at(i + 1));
    }

    // <0> is cropVision, <1> is affineMatrix
    auto cropVisionFrameAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(visionFrame, faceLandmark5_68,
                                                                            warpTemplate,
                                                                            cv::Size(112, 112));

    std::vector<cv::Mat> bgrChannels(3);
    split(std::get<0>(*cropVisionFrameAndAffineMat), bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 127.5, -1.0);
    }

    const int imageArea = this->m_inputHeight * this->m_inputWidth;
    std::vector<float> inputData;
    inputData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputData.data(), (float *)bgrChannels[2].data, singleChnSize);
    memcpy(inputData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(inputData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);
    return inputData;
}
} // namespace Ffc
