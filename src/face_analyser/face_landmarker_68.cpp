/**
 ******************************************************************************
 * @file           : face_landmarker_68.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#include "face_landmarker_68.h"

namespace Ffc {
FaceLandmarker68::FaceLandmarker68(const std::shared_ptr<Ort::Env> &env,
                                   const std::shared_ptr<const nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = "./models/2dfan4.onnx";
    // 如果 modelPath不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::downloadFileFromURL(m_modelsInfoJson->at("faceAnalyserModels").at("face_landmarker_68").at("url"),
                                                               "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }

    this->createSession(modelPath);
}

std::shared_ptr<std::tuple<Typing::FaceLandmark, Typing::Score>>
FaceLandmarker68::detect(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    this->preProcess(visionFrame, boundingBox);

    std::vector<int64_t> inputImgShape = {1, 3, this->m_inputHeight, this->m_inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(this->m_memoryInfo, this->m_inputData.data(), this->m_inputData.size(), inputImgShape.data(), inputImgShape.size());

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = this->m_session->Run(runOptions, this->m_inputNames.data(), &inputTensor, 1, this->m_outputNames.data(), this->m_outputNames.size());

    float *pdata = ortOutputs[0].GetTensorMutableData<float>(); /// 形状是(1, 68, 3), 每一行的长度是3，表示一个关键点坐标x,y和置信度
    const int numPoints = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape()[1];
    std::vector<cv::Point2f> faceLandmark68(numPoints);
    std::vector<Typing::Score> scores(numPoints);
    for (int i = 0; i < numPoints; i++) {
        float x = pdata[i * 3] / 64.0 * 256.0;
        float y = pdata[i * 3 + 1] / 64.0 * 256.0;
        float score = pdata[i * 3 + 2];
        faceLandmark68[i] = cv::Point2f(x, y);
        scores[i] = score;
    }
    cv::transform(faceLandmark68, faceLandmark68, this->m_invAffineMatrix);

    float sum = 0.0;
    for (int i = 0; i < numPoints; i++) {
        sum += scores[i];
    }
    float meanScore = sum / (float)numPoints;
    return std::make_shared<std::tuple<Typing::FaceLandmark,
                                       Typing::Score>>(std::make_tuple(faceLandmark68, meanScore));
}

void FaceLandmarker68::preProcess(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    m_inputHeight = m_inputNodeDims[0][2];
    m_inputWidth = m_inputNodeDims[0][3];

    float sub_max = std::max(boundingBox.xmax - boundingBox.xmin, boundingBox.ymax - boundingBox.ymin);
    const float scale = 195.f / sub_max;
    const std::vector<float> translation = {(256.f - (boundingBox.xmax + boundingBox.xmin) * scale) * 0.5f, (256.f - (boundingBox.ymax + boundingBox.ymin) * scale) * 0.5f};

    auto cropVisionFrameAndAffineMat = FaceHelper::warpFaceByTranslation(visionFrame, translation,
                                                                         scale, cv::Size{256, 256});
    cv::Mat cropImg = std::get<0>(*cropVisionFrameAndAffineMat);
    cv::Mat affineMatrix = std::get<1>(*cropVisionFrameAndAffineMat);
    cv::invertAffineTransform(affineMatrix, this->m_invAffineMatrix);

    std::vector<cv::Mat> bgrChannels(3);
    split(cropImg, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.0);
    }

    const int imageArea = this->m_inputHeight * this->m_inputWidth;
    this->m_inputData.resize(3 * imageArea);
    const size_t singleChnSize = imageArea * sizeof(float);
    memcpy(this->m_inputData.data(), (float *)bgrChannels[0].data, singleChnSize);
    memcpy(this->m_inputData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(this->m_inputData.data() + imageArea * 2, (float *)bgrChannels[2].data, singleChnSize);
}
} // namespace Ffc