/**
 ******************************************************************************
 * @file           : face_masker.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-8
 ******************************************************************************
 */

#include "face_masker.h"
#include <iostream>
#include <fstream>

namespace Ffc {
std::shared_ptr<cv::Mat>
FaceMasker::createStaticBoxMask(const cv::Size &cropSize, const float &faceMaskBlur,
                                const Typing::Padding &faceMaskPadding) {
    int blurAmount = static_cast<int>(cropSize.width * 0.5 * faceMaskBlur);
    int blurArea = std::max(blurAmount / 2, 1);

    cv::Mat boxMask(cropSize, CV_32F, cv::Scalar(1.0f));

    int paddingTop = std::max(blurArea, static_cast<int>(cropSize.height * std::get<0>(faceMaskPadding) / 100.0));
    int paddingBottom = std::max(blurArea, static_cast<int>(cropSize.height * std::get<2>(faceMaskPadding) / 100.0));
    int paddingLeft = std::max(blurArea, static_cast<int>(cropSize.width * std::get<3>(faceMaskPadding) / 100.0));
    int paddingRight = std::max(blurArea, static_cast<int>(cropSize.width * std::get<1>(faceMaskPadding) / 100.0));

    boxMask(cv::Range(0, paddingTop), cv::Range::all()) = 0;
    boxMask(cv::Range(cropSize.height - paddingBottom, cropSize.height), cv::Range::all()) = 0;
    boxMask(cv::Range::all(), cv::Range(0, paddingLeft)) = 0;
    boxMask(cv::Range::all(), cv::Range(cropSize.width - paddingRight, cropSize.width)) = 0;

    if (blurAmount > 0) {
        cv::GaussianBlur(boxMask, boxMask, cv::Size(0, 0), blurAmount * 0.25);
    }

    return std::make_shared<cv::Mat>(std::move(boxMask));
}

FaceMasker::FaceMasker(const std::shared_ptr<Ort::Env> &env,
                       const std::shared_ptr<nlohmann::json> &modelsInfoJson) :
    OrtSession(env) {
    m_modelsJson = modelsInfoJson;
}

std::shared_ptr<cv::Mat> FaceMasker::createOcclusionMask(const Typing::VisionFrame &cropVisionFrame) {
    std::string modelPath = m_modelsJson->at("faceMaskerModels").at("face_occluder").at("path");
    // Todo 检查model是否存在，若不存在则下载

    this->createSession(modelPath);
    const int inputHeight = m_inputNodeDims[0][1];
    const int inputWidth = m_inputNodeDims[0][2];

    cv::Mat inputImage = cropVisionFrame.clone();
    cv::resize(cropVisionFrame, inputImage, cv::Size(inputHeight, inputWidth));
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);
    for (int i = 0; i < 3; i++) {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / 255.0);
    }
    const int imageArea = inputHeight * inputWidth;
    std::vector<float> inputImageData(3 * imageArea);
    
    // 妈的，傻逼输入形状(1, 256, 256, 3), 害得老子排查了一天半的bug, 老子服了
    int k = 0;
    for (int i = 0; i < inputHeight; ++i) {
        for (int j = 0; j < inputWidth; ++j) {
            inputImageData.at(k) = bgrChannels[2].at<float>(i, j);
            inputImageData.at(k + 1) = bgrChannels[1].at<float>(i, j);
            inputImageData.at(k + 2) = bgrChannels[0].at<float>(i, j);
            k += 3;
        }
    }

    std::vector<int64_t> inputImageShape{1, inputHeight, inputWidth, 3};

    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, inputImageData.data(),
                                                              inputImageData.size(),
                                                              inputImageShape.data(),
                                                              inputImageShape.size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensors = m_session->Run(runOptions, m_inputNames.data(),
                                                           inputTensors.data(), inputTensors.size(),
                                                           m_outputNames.data(), m_outputNames.size());

    float *pdata = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outsShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int outputHeight = outsShape[1];
    const int outputWidth = outsShape[2];
    
    cv::Mat mask(outputHeight, outputWidth, CV_32FC1, pdata);
    mask.setTo(0, mask < 0);
    mask.setTo(1, mask > 1);
    
    cv::resize(mask, mask, cropVisionFrame.size());
    mask.setTo(0, mask < 0);
    mask.setTo(1, mask > 1);

    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 5);
    mask.setTo(0, mask < 0);
    mask.setTo(1, mask > 1);

    mask.setTo(0.5, mask < 0.5);
    mask.setTo(1, mask > 1);
    mask = (mask - 0.5) * 2;
    return std::make_shared<cv::Mat>(std::move(mask));
}
} // namespace Ffc