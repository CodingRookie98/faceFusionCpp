/**
 ******************************************************************************
 * @file           : face_detector_gender_age.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-15
 ******************************************************************************
 */

#include "face_detector_gender_age.h"

namespace Ffc {
FaceDetectorGenderAge::FaceDetectorGenderAge(const std::shared_ptr<Ort::Env> &env) :
    OrtSession(env) {
    this->createSession("./models/gender_age.onnx");
}

std::shared_ptr<std::tuple<int, int>> FaceDetectorGenderAge::detect(const VisionFrame &visionFrame, const BoundingBox &boundingBox) {
    float boundingBoxW = boundingBox.xmax - boundingBox.xmin;
    float boundingBoxH = boundingBox.ymax - boundingBox.ymin;
    float maxSide = std::max(boundingBoxW, boundingBoxH);
    float scale = (float)64 / (float)maxSide;
    std::vector<float> translation;
    translation.emplace_back(48 - scale * (boundingBox.xmin + boundingBox.xmax) * 0.5);
    translation.emplace_back(48 - scale * (boundingBox.ymin + boundingBox.ymax) * 0.5);
    auto cropVisionAndAffineMat = FaceHelper::warpFaceByTranslation(visionFrame, translation,
                                                                    scale, cv::Size(96, 96));

    int inputHeight = m_inputNodeDims[0][2];
    int inputWidth = m_inputNodeDims[0][3];

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
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo,
                                                              inputImageData.data(), inputImageData.size(),
                                                              inputTensorShape.data(), inputTensorShape.size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensor = m_session->Run(runOptions, m_inputNames.data(),
                                                                   inputTensors.data(), inputTensors.size(),
                                                                   m_outputNames.data(), m_outputNames.size());
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