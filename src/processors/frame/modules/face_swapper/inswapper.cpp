/**
 ******************************************************************************
 * @file           : inswapper.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-9
 ******************************************************************************
 */

#include "inswapper.h"

namespace Ffc {
Inswapper::Inswapper(const std::shared_ptr<Ort::Env> &env) :
    FaceSwapperSession(env, (Globals::faceSwapperModel == Globals::EnumFaceSwapperModel::InSwapper_128) ? "./models/inswapper_128.onnx" : "./models/inswapper_128_fp16.onnx") {
    std::string modelPath = Globals::faceSwapperModel == Globals::EnumFaceSwapperModel::InSwapper_128 ? "./models/inswapper_128.onnx" : "./models/inswapper_128_fp16.onnx";

    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];

    // Load ONNX model as a protobuf message
    onnx::ModelProto model_proto;
    std::ifstream input(modelPath, std::ios::binary);
    if (!model_proto.ParseFromIstream(&input)) {
        throw std::runtime_error("Failed to load model.");
    }

    // Access the initializer
    const onnx::TensorProto &initializer = model_proto.graph().initializer(model_proto.graph().initializer_size() - 1);
    // Convert initializer to an array
    m_initializerArray.assign(initializer.float_data().begin(), initializer.float_data().end());
}

std::shared_ptr<Typing::VisionFrame> Inswapper::applySwap(const Face &sourceFace, const Face &targetFace, const VisionFrame &targetFrame) {
    auto croppedTargetFrameAndAffineMat = Ffc::Inswapper::getCropVisionFrameAndAffineMat(
        targetFrame, targetFace.faceLandMark5_68, m_warpTemplate, m_size);
    auto preparedTargetFrameBGR = Ffc::Inswapper::prepareCropVisionFrame(
        std::get<0>(*croppedTargetFrameAndAffineMat), m_mean, m_standardDeviation);
    auto cropMaskList = Ffc::Inswapper::getCropMaskList(std::get<0>(*croppedTargetFrameAndAffineMat),
                                                        std::get<0>(*croppedTargetFrameAndAffineMat).size(),
                                                        Globals::faceMaskBlur, Globals::faceMaskPadding);

    cv::Mat bgrImage = preparedTargetFrameBGR->clone();
    std::vector<cv::Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);
    const int imageArea = this->m_inputHeight * this->m_inputWidth;
    this->m_inputImageData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(this->m_inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize);                 // R
    memcpy(this->m_inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);     // G
    memcpy(this->m_inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize); // B

    double norm = cv::norm(sourceFace.embedding, cv::NORM_L2);
    m_inputEmbeddingData.resize(m_lenFeature);
    for (int i = 0; i < m_lenFeature; ++i) {
        double sum = 0.0f;
        for (int j = 0; j < m_lenFeature; ++j) {
            sum += sourceFace.embedding.at(j)
                   * m_initializerArray.at(j * m_lenFeature + i);
        }
        m_inputEmbeddingData.at(i) = (float)(sum / norm);
    }

    // Create input tensors
    std::vector<Ort::Value> inputTensors;
    std::vector<int64_t> inputImageShape = {1, 3, m_inputHeight, m_inputWidth};
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, m_inputImageData.data(), m_inputImageData.size(), inputImageShape.data(), inputImageShape.size()));
    std::vector<int64_t> inputEmbeddingShape = {1, m_lenFeature};
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, m_inputEmbeddingData.data(), m_inputEmbeddingData.size(), inputEmbeddingShape.data(), inputEmbeddingShape.size()));

    Ort::RunOptions runOptions;
    auto outputTensor = m_session->Run(runOptions, m_inputNames.data(), inputTensors.data(), inputTensors.size(), m_outputNames.data(), m_outputNames.size());

    float *pdata = outputTensor[0].GetTensorMutableData<float>();
    std::vector<int64_t> outsShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
    const int outputHeight = outsShape[2];
    const int outputWidth = outsShape[3];

    const int channelStep = outputHeight * outputWidth;
    std::vector<cv::Mat> channelMats(3);
    // Create matrices for each channel and scale/clamp values
    channelMats[2] = cv::Mat(outputHeight, outputWidth, CV_32FC1, pdata);                   // R
    channelMats[1] = cv::Mat(outputHeight, outputWidth, CV_32FC1, pdata + channelStep);     // G
    channelMats[0] = cv::Mat(outputHeight, outputWidth, CV_32FC1, pdata + 2 * channelStep); // B
    for (auto &mat : channelMats) {
        mat *= 255.f;
        mat.setTo(0, mat < 0);
        mat.setTo(255, mat > 255);
    }
    // Merge the channels into a single matrix
    cv::Mat resultMat;
    cv::merge(channelMats, resultMat);

    for (auto &cropMask : *cropMaskList) {
        cropMask.setTo(0, cropMask < 0);
        cropMask.setTo(1, cropMask > 1);
    }

    cv::Mat dstImage = FaceHelper::pasteBack(targetFrame, resultMat, cropMaskList->front(),
                                             std::get<1>(*croppedTargetFrameAndAffineMat));

    return std::make_shared<Typing::VisionFrame>(dstImage);
}
} // namespace Ffc