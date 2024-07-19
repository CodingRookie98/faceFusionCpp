/**
 ******************************************************************************
 * @file           : face_landmarker_68_5.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#include "face_landmarker_68_5.h"

namespace Ffc {
FaceLandmarker68_5::FaceLandmarker68_5(const std::shared_ptr<Ort::Env> &env,
                                       const std::shared_ptr<const nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = "./models/face_landmarker_68_5.onnx";
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

void FaceLandmarker68_5::preProcess(const FaceLandmark &faceLandmark5) {
    m_inputHeight = m_inputNodeDims[0][1];
    m_inputWidth = m_inputNodeDims[0][2];
    Ffc::Typing::FaceLandmark landmark5 = faceLandmark5;

    std::vector<float> fVec = m_modelsInfoJson->at("faceHelper").at("warpTemplate").at("ffhq_512").get<std::vector<float>>();
    std::vector<cv::Point2f> warpTemplate;
    for (int i = 0; i < fVec.size(); i += 2) {
        warpTemplate.emplace_back(fVec.at(i), fVec.at(i + 1));
    }

    m_affineMatrix = Ffc::FaceHelper::estimateMatrixByFaceLandmark5(landmark5, warpTemplate, cv::Size(1, 1));
    cv::transform(landmark5, landmark5, m_affineMatrix);

    for (const auto &point : landmark5) {
        m_inputTensorData.emplace_back(point.x);
        m_inputTensorData.emplace_back(point.y);
    }
}

std::shared_ptr<Typing::FaceLandmark> FaceLandmarker68_5::detect(const FaceLandmark &faceLandmark5) {
    this->preProcess(faceLandmark5);

    std::vector<int64_t> inputShape{1, this->m_inputHeight, this->m_inputWidth};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(this->m_memoryInfo,
                                                             m_inputTensorData.data(),
                                                             m_inputTensorData.size(), inputShape.data(),
                                                             inputShape.size());
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensor = this->m_session->Run(runOptions, this->m_inputNames.data(),
                                                                &inputTensor, 1, this->m_outputNames.data(),
                                                                m_outputNames.size());
    auto *pData = outputTensor[0].GetTensorMutableData<float>(); // shape(1, 68, 2);
    auto faceLandMark68_5 = std::make_shared<Typing::FaceLandmark>();
    for (int i = 0; i < 68; ++i) {
        faceLandMark68_5->emplace_back(pData[i * 2], pData[i * 2 + 1]);
    }

    // 将result转换为Mat类型，并确保形状为 (68, 2)
    cv::Mat resultMat(*faceLandMark68_5);

    // 进行仿射变换
    cv::Mat transformedMat;
    cv::Mat affineMatrixInv;
    cv::invertAffineTransform(m_affineMatrix, affineMatrixInv);
    cv::transform(resultMat, transformedMat, affineMatrixInv);

    // 更新result
    faceLandMark68_5->clear();
    for (int i = 0; i < transformedMat.rows; ++i) {
        faceLandMark68_5->emplace_back(transformedMat.at<cv::Point2f>(i));
    }
    return faceLandMark68_5;
}
} // namespace Ffc
