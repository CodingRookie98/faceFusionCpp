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
FaceRecognizerArcW600kR50::FaceRecognizerArcW600kR50(const std::shared_ptr<Ort::Env> &env, const std::shared_ptr<nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = "./models/arcface_w600k_r50.onnx";
    // 如果 modelPath不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::downloadFileFromURL(m_modelsInfoJson->at("faceAnalyserModels").at("face_recognizer_arcface_uniface").at("url"),
                                                               "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    this->createSession(modelPath);
}

std::shared_ptr<std::tuple<Typing::Embedding, Typing::Embedding>>
FaceRecognizerArcW600kR50::recognize(const Typing::VisionFrame &visionFrame,
                                     const Typing::FaceLandmark &faceLandmark5) {
    this->preProcess(visionFrame, faceLandmark5);
    std::vector<int64_t> inputImgShape = {1, 3, this->m_inputHeight, this->m_inputWidth};
    Ort::Value inputTensor             = Ort::Value::CreateTensor<float>(this->m_memoryInfo, this->m_inputData.data(),
                                                             this->m_inputData.size(),
                                                             inputImgShape.data(), inputImgShape.size());

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = this->m_session->Run(runOptions, this->m_inputNames.data(), &inputTensor, 1, this->m_outputNames.data(), m_outputNames.size());

    float *pdata         = ortOutputs[0].GetTensorMutableData<float>(); /// 形状是(1, 512)
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

void FaceRecognizerArcW600kR50::preProcess(const Typing::VisionFrame &visionFrame,
                                           const Typing::FaceLandmark &faceLandmark5_68) {
    m_inputWidth  = (int)m_inputNodeDims[0][2];
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
    this->m_inputData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(this->m_inputData.data(), (float *)bgrChannels[2].data, singleChnSize);
    memcpy(this->m_inputData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(this->m_inputData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);
}
} // namespace Ffc
