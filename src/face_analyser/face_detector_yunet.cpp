/**
 ******************************************************************************
 * @file           : face_detector_yunet.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-16
 ******************************************************************************
 */

#include "face_detector_yunet.h"

namespace Ffc {
FaceDetectorYunet::FaceDetectorYunet(const std::shared_ptr<Ort::Env> &env,
                                     const std::shared_ptr<const nlohmann::json> &modelsInfoJson) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson) {
    std::string modelPath = m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_yunet").at("path");

    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceAnalyserModels").at("face_detector_yunet").at("url"),
                                                               "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    m_faceDetectorYN = cv::FaceDetectorYN::create(modelPath, "", cv::Size(0, 0));
}

std::shared_ptr<std::tuple<std::vector<Typing::BoundingBox>,
                           std::vector<Typing::FaceLandmark>,
                           std::vector<Typing::Score>>>
FaceDetectorYunet::detect(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize,
                          const float &scoreThreshold) {
    preProcess(visionFrame, faceDetectorSize);

    m_faceDetectorYN->setInputSize(cv::Size(m_inputWidth, m_inputHeight));
    m_faceDetectorYN->setScoreThreshold(scoreThreshold);
    cv::Mat output;
    m_faceDetectorYN->detect(m_inputVisionFrame, output);

    std::vector<Typing::BoundingBox> resultBoundingBoxes;
    std::vector<Typing::FaceLandmark> resultFaceLandmarks;
    std::vector<Typing::Score> resultScores;
    for (size_t i = 0; i < output.rows; ++i) {
        Typing::BoundingBox tempBbox;
        tempBbox.xmin = output.at<float>(i, 0) * m_ratioWidth;
        tempBbox.ymin = output.at<float>(i, 1) * m_ratioHeight;
        tempBbox.xmax = (output.at<float>(i, 0) + output.at<float>(i, 2)) * m_ratioWidth;
        tempBbox.ymax = (output.at<float>(i, 1) + output.at<float>(i, 3)) * m_ratioWidth;
        resultBoundingBoxes.emplace_back(tempBbox);

        Typing::FaceLandmark tempLandmark;
        for (size_t j = 4; j < 14; j += 2) {
            cv::Point2f tempPoint;
            tempPoint.x = output.at<float>(i, j) * m_ratioWidth;
            tempPoint.y = output.at<float>(i, j + 1) * m_ratioHeight;
            tempLandmark.emplace_back(tempPoint);
        }
        resultFaceLandmarks.emplace_back(tempLandmark);

        resultScores.emplace_back(output.at<float>(i, 14));
    }

    return std::make_shared<std::tuple<std::vector<Typing::BoundingBox>,
                                       std::vector<Typing::FaceLandmark>,
                                       std::vector<Typing::Score>>>(
        std::make_tuple(resultBoundingBoxes, resultFaceLandmarks, resultScores));
}

void FaceDetectorYunet::preProcess(const VisionFrame &visionFrame, const cv::Size &faceDetectorSize) {
    const int faceDetectorHeight = faceDetectorSize.height;
    const int faceDetectorWidth = faceDetectorSize.width;

    auto tempVisionFrame = Vision::resizeFrameResolution(visionFrame, cv::Size(faceDetectorWidth, faceDetectorHeight));
    m_ratioHeight = (float)visionFrame.rows / (float)tempVisionFrame.rows;
    m_ratioWidth = (float)visionFrame.cols / (float)tempVisionFrame.cols;
    m_inputHeight = tempVisionFrame.rows;
    m_inputWidth = tempVisionFrame.cols;
    m_inputVisionFrame = tempVisionFrame.clone();
}
} // namespace Ffc