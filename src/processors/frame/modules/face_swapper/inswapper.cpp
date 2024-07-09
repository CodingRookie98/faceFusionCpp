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

    m_inputHeight = m_inputNodeDims[0][2];
    m_inputWidth = m_inputNodeDims[0][3];

    //    // Load ONNX model as a protobuf message
    //    onnx::ModelProto model_proto;
    //    std::ifstream input(modelPath, std::ios::binary);
    //    if (!model_proto.ParseFromIstream(&input)) {
    //        throw std::runtime_error("Failed to load model.");
    //    }
    //
    //    // Access the initializer
    //    const onnx::TensorProto &initializer = model_proto.graph().initializer(model_proto.graph().initializer_size() - 1);
    //
    //    // Convert initializer to an array
    //    m_initializerArray.assign(initializer.float_data().begin(), initializer.float_data().end());

    const int length = this->m_lenFeature * this->m_lenFeature;
    this->m_initializerArray.resize(length);
    FILE *fp = fopen("model_matrix.bin", "rb");
    for (int i = 0; i < length; ++i) {
        float tmp = 0;
        fread(&tmp, sizeof(float), 1, fp);
        this->m_initializerArray.at(i) = tmp;
    }
    fclose(fp); // 关闭文件
}

std::shared_ptr<Typing::VisionFrame> Inswapper::applySwap(const Face &sourceFace, const Face &targetFace, const VisionFrame &targetFrame) {
    std::vector<cv::Point2f> normed_template;
    normed_template.emplace_back(cv::Point2f(46.29459968, 51.69629952));
    normed_template.emplace_back(cv::Point2f(81.53180032, 51.50140032));
    normed_template.emplace_back(cv::Point2f(64.02519936, 71.73660032));
    normed_template.emplace_back(cv::Point2f(49.54930048, 92.36550016));
    normed_template.emplace_back(cv::Point2f(78.72989952, 92.20409984));
    auto croppedTargetFrameAndAffineMat = Ffc::Inswapper::getCropVisionFrameAndAffineMat(
        targetFrame, targetFace.faceLandMark5_68, normed_template, m_size);
    //    auto croppedTargetFrameAndAffineMat = Ffc::Inswapper::getCropVisionFrameAndAffineMat(
    //        targetFrame, targetFace.faceLandMark5_68, m_warpTemplate, m_size);
    // T
    //    auto preparedTargetFrameRGB = Ffc::Inswapper::prepareCropVisionFrame(
    //        std::get<0>(*croppedTargetFrameAndAffineMat), m_mean, m_standardDeviation); // F
    auto cropMaskList = Ffc::Inswapper::getCropMaskList(std::get<0>(*croppedTargetFrameAndAffineMat),
                                                        std::get<0>(*croppedTargetFrameAndAffineMat).size(),
                                                        Globals::faceMaskBlur, Globals::faceMaskPadding);

    cv::Mat bgrImage = std::get<0>(*croppedTargetFrameAndAffineMat).clone();
    std::vector<cv::Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels.at(c), CV_32FC1, 1 / (255.0 * m_standardDeviation.at(c)),
                                 -m_mean.at(c) / (float)m_standardDeviation.at(c));
    }

    const int image_area = this->m_inputHeight * this->m_inputWidth;
    this->m_inputImageData.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->m_inputImageData.data(), (float *)bgrChannels[2].data, single_chn_size); /// rgb顺序
    memcpy(this->m_inputImageData.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->m_inputImageData.data() + image_area * 2, (float *)bgrChannels[0].data, single_chn_size);

    //    auto tempEmbedding = new Typing::Embedding(sourceFace.embedding);
    //    cv::Mat embedding = cv::Mat(1, m_lenFeature, CV_32F, tempEmbedding->data());
    //    delete tempEmbedding;
    //    cv::Mat initializer = cv::Mat(m_lenFeature, m_lenFeature, CV_32F, m_initializerArray.data());
    //    cv::Mat result = embedding * initializer;
    //    result /= cv::norm(embedding, cv::NORM_L2);
    //    m_inputEmbeddingData.resize(m_lenFeature);
    //    std::memcpy(m_inputEmbeddingData.data(), result.data, result.total() * sizeof(float));

    double norm = cv::norm(sourceFace.embedding, cv::NORM_L2);
    m_inputEmbeddingData.resize(m_lenFeature);
    for (int i = 0; i < m_lenFeature; ++i) {
        double sum = 0.0f;
        for (int j = 0; j < m_lenFeature; ++j) {
            sum += static_cast<double>(sourceFace.embedding.at(j)) *
                   static_cast<double>(m_initializerArray.at(j * m_lenFeature + i));
        }
        m_inputEmbeddingData.at(i) = static_cast<float>(sum / norm);
    }

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
    cv::Mat rmat(outputHeight, outputWidth, CV_32FC1, pdata);
    cv::Mat gmat(outputHeight, outputWidth, CV_32FC1, pdata + channelStep);
    cv::Mat bmat(outputHeight, outputWidth, CV_32FC1, pdata + 2 * channelStep);
    rmat *= 255.f;
    gmat *= 255.f;
    bmat *= 255.f;
    rmat.setTo(0, rmat < 0);
    rmat.setTo(255, rmat > 255);
    gmat.setTo(0, gmat < 0);
    gmat.setTo(255, gmat > 255);
    bmat.setTo(0, bmat < 0);
    bmat.setTo(255, bmat > 255);

    std::vector<cv::Mat> channel_mats(3);
    channel_mats[0] = bmat;
    channel_mats[1] = gmat;
    channel_mats[2] = rmat;
    cv::Mat resultMat;
    merge(channel_mats, resultMat);
    //    // Step 1: Create a cv::Mat object with the data
    //    cv::Mat rgbImage(outputHeight, outputWidth, CV_32FC3, pdata);
    //    // Step 2: Scale and clamp data to [0, 255]
    //    cv::Mat scaledRGB;
    //    rgbImage.convertTo(scaledRGB, CV_32FC3, 255.0);
    //    cv::threshold(scaledRGB, scaledRGB, 255.0, 255.0, cv::THRESH_TRUNC);
    //    cv::threshold(scaledRGB, scaledRGB, 0.0, 0.0, cv::THRESH_TOZERO);
    //    // Step 3: Convert from RGB to BGR
    //    cv::Mat bgrImage;
    //    cv::cvtColor(scaledRGB, bgrImage, cv::COLOR_RGB2BGR);

    for (auto &cropMask : *cropMaskList) {
        cropMask.setTo(0, cropMask < 0);
        cropMask.setTo(1, cropMask > 1);
    }

    cv::Mat dstImage = FaceHelper::pasteBack(targetFrame, resultMat, cropMaskList->front(),
                                             std::get<1>(*croppedTargetFrameAndAffineMat));

    return std::make_shared<Typing::VisionFrame>(dstImage);
}
} // namespace Ffc