/**
 ******************************************************************************
 * @file           : face_detector_scrfd.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-16
 ******************************************************************************
 */

#include "face_detector_scrfd.h"

namespace Ffc {

FaceDetectorScrfd::FaceDetectorScrfd(const std::shared_ptr<Ort::Env> &env,
                                     const std::shared_ptr<const nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_scrfd").at("path");

    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_scrfd").at("url"),
                                                               "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    this->createSession(modelPath);
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceDetectorScrfd::detect(const Typing::VisionFrame &visionFrame,
                          const cv::Size &faceDetectorSize, const float &detectorScore) {
    preProcess(visionFrame, faceDetectorSize);

    std::vector<int64_t> inputImgShape = {1, 3, faceDetectorSize.height, faceDetectorSize.width};
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(m_memoryInfo, m_inputData.data(), m_inputData.size(), inputImgShape.data(), inputImgShape.size());
    Ort::RunOptions runOptions;
    std::vector<Ort::Value> ortOutputs = m_session->Run(runOptions, m_inputNames.data(), &inputTensor, 1, m_outputNames.data(), m_outputNames.size());

    std::vector<Typing::BoundingBox> resultBoundingBoxes;
    std::vector<Typing::FaceLandmark> resultFaceLandmarks;
    std::vector<Typing::Score> resultScores;

    for (size_t index = 0; index < m_featureStrides.size(); ++index) {
        int featureStride = m_featureStrides[index];
        std::vector<int> keepIndices;
        int size = ortOutputs[index].GetTensorTypeAndShapeInfo().GetShape()[0];
        float *pdataScore = ortOutputs[index].GetTensorMutableData<float>();
        for (size_t j = 0; j < size; ++j) {
            float tempScore = *(pdataScore + j);
            if (tempScore >= detectorScore) {
                keepIndices.emplace_back(j);
            }
        }

        if (!keepIndices.empty()) {
            int strideHeight = std::floor(faceDetectorSize.height / featureStride);
            int strideWidth = std::floor(faceDetectorSize.width / featureStride);

            std::vector<std::array<int, 2>> anchors = FaceHelper::createStaticAnchors(m_featureStrides[index],
                                                                                      m_anchorTotal,
                                                                                      strideHeight,
                                                                                      strideWidth);

            std::vector<Typing::BoundingBox> boundingBoxesRaw;
            std::vector<Typing::FaceLandmark> faceLandmarksRaw;

            float *pdataBbox = ortOutputs[index + m_featureMapChannel].GetTensorMutableData<float>();
            float *pdataLandmark = ortOutputs[index + 2 * m_featureMapChannel].GetTensorMutableData<float>();
            float *pdataScore = ortOutputs[index].GetTensorMutableData<float>();

            size_t pdataBboxSize = ortOutputs[index + m_featureMapChannel].GetTensorTypeAndShapeInfo().GetShape()[0]
                                   * ortOutputs[index + m_featureMapChannel].GetTensorTypeAndShapeInfo().GetShape()[1];
            for (size_t k = 0; k < pdataBboxSize; k += 4) {
                Typing::BoundingBox tempBbox;
                tempBbox.xmin = *(pdataBbox + k) * featureStride;
                tempBbox.ymin = *(pdataBbox + k + 1) * featureStride;
                tempBbox.xmax = *(pdataBbox + k + 2) * featureStride;
                tempBbox.ymax = *(pdataBbox + k + 3) * featureStride;
                boundingBoxesRaw.emplace_back(tempBbox);
            }

            size_t pdataLandmarkSize = ortOutputs[index + 2 * m_featureMapChannel].GetTensorTypeAndShapeInfo().GetShape()[0]
                                       * ortOutputs[index + 2 * m_featureMapChannel].GetTensorTypeAndShapeInfo().GetShape()[1];
            for (size_t k = 0; k < pdataLandmarkSize; k += 10) {
                Typing::FaceLandmark tempLandmark;
                tempLandmark.emplace_back(cv::Point2f(*(pdataLandmark + k), *(pdataLandmark + k + 1)));
                tempLandmark.emplace_back(cv::Point2f(*(pdataLandmark + k + 2), *(pdataLandmark + k + 3)));
                tempLandmark.emplace_back(cv::Point2f(*(pdataLandmark + k + 4), *(pdataLandmark + k + 5)));
                tempLandmark.emplace_back(cv::Point2f(*(pdataLandmark + k + 6), *(pdataLandmark + k + 7)));
                tempLandmark.emplace_back(cv::Point2f(*(pdataLandmark + k + 8), *(pdataLandmark + k + 9)));
                for (auto &point : tempLandmark) {
                    point.x *= featureStride;
                    point.y *= featureStride;
                }
                faceLandmarksRaw.emplace_back(tempLandmark);
            }

            for (const auto &keepIndex : keepIndices) {
                auto tempBbox = FaceHelper::distance2BoundingBox(anchors[keepIndex], boundingBoxesRaw[keepIndex]);
                tempBbox->xmin *= m_ratioWidth;
                tempBbox->ymin *= m_ratioHeight;
                tempBbox->xmax *= m_ratioWidth;
                tempBbox->ymax *= m_ratioHeight;
                resultBoundingBoxes.emplace_back(*tempBbox);

                auto tempLandmark = FaceHelper::distance2FaceLandmark5(anchors[keepIndex], faceLandmarksRaw[keepIndex]);
                for (auto &point : *tempLandmark) {
                    point.x *= m_ratioWidth;
                    point.y *= m_ratioHeight;
                }
                resultFaceLandmarks.emplace_back(*tempLandmark);

                resultScores.emplace_back(Typing::Score{*(pdataScore + keepIndex)});
            }
        }
    }

    return std::make_shared<std::tuple<std::vector<Typing::BoundingBox>,
                                       std::vector<Typing::FaceLandmark>,
                                       std::vector<Typing::Score>>>(resultBoundingBoxes, resultFaceLandmarks, resultScores);
}

void FaceDetectorScrfd::preProcess(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    const int faceDetectorHeight = faceDetectorSize.height;
    const int faceDetectorWidth = faceDetectorSize.width;
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];

    auto tempVisionFrame = Vision::resizeFrameResolution(visionFrame, cv::Size(faceDetectorWidth, faceDetectorHeight));
    m_ratioHeight = (float)visionFrame.rows / (float)tempVisionFrame.rows;
    m_ratioWidth = (float)visionFrame.cols / (float)tempVisionFrame.cols;

    // 创建一个指定尺寸的全零矩阵
    cv::Mat detectVisionFrame = cv::Mat::zeros(faceDetectorHeight, faceDetectorWidth, CV_32FC3);
    // 将输入的图像帧复制到全零矩阵的左上角
    tempVisionFrame.copyTo(detectVisionFrame(cv::Rect(0, 0, tempVisionFrame.cols, tempVisionFrame.rows)));

    std::vector<cv::Mat> bgrChannels(3);
    cv::split(detectVisionFrame, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int imageArea = faceDetectorHeight * faceDetectorWidth;
    m_inputData.resize(3 * imageArea);
    const size_t singleChnSize = imageArea * sizeof(float);
    memcpy(m_inputData.data(), (float *)bgrChannels[0].data, singleChnSize);
    memcpy(m_inputData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(m_inputData.data() + imageArea * 2, (float *)bgrChannels[2].data, singleChnSize);
}
} // namespace Ffc