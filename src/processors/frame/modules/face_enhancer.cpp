/**
 ******************************************************************************
 * @file           : face_enhancer.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-11
 ******************************************************************************
 */

#include "face_enhancer.h"
// Todo
#include <iostream>
#include <fstream>

namespace Ffc {
FaceEnhancer::FaceEnhancer(const std::shared_ptr<Ort::Env> &env) :
    m_env(env) {
    m_sessionOptions = Ort::SessionOptions();
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    m_cudaProviderOptions = std::make_shared<OrtCUDAProviderOptions>();
}

void FaceEnhancer::init() {
    if (m_faceEnhancerModel != nullptr && *m_faceEnhancerModel == Globals::faceEnhancerModel) {
        return;
    }

    m_faceEnhancerModel = std::make_shared<Globals::EnumFaceEnhancerModel>(Globals::faceEnhancerModel);
    m_cudaProviderOptions->device_id = 0;
    m_sessionOptions.AppendExecutionProvider_CUDA(*m_cudaProviderOptions);

    // Todo
    m_modelsJson = std::make_shared<nlohmann::json>();
    std::ifstream file("./modelsInfo.json");
    if (file.is_open()) {
        file >> *m_modelsJson;
        file.close();
    }

    // Todo
    switch (Globals::faceEnhancerModel) {
    case Globals::FE_Gfpgan_14:
        m_modelName = "gfpgan_1.4";
        break;
    case Globals::FE_CodeFormer:
        m_modelName = "codeformer";
        break;
    default:
        m_modelName = "gfpgan_1.4";
        break;
    }
    std::string modelPath = m_modelsJson->at("faceEnhancerModels").at(m_modelName).at("path");

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    // windows
    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    m_session = std::make_shared<Ort::Session>(*m_env, wideModelPath.c_str(), m_sessionOptions);
#else
    // linux
    m_session = std::make_shared<Ort::Session>(Ort::Session(*m_env, modelPath.c_str(), m_sessionOptions));
#endif

    size_t numInputNodes = m_session->GetInputCount();
    size_t numOutputNodes = m_session->GetOutputCount();

    m_inputNames.reserve(numInputNodes);
    m_outputNames.reserve(numOutputNodes);
    m_inputNamesPtrs.reserve(numInputNodes);
    m_outputNamesPtrs.reserve(numOutputNodes);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < numInputNodes; i++) {
        m_inputNamesPtrs.push_back(std::move(m_session->GetInputNameAllocated(i, allocator)));
        m_inputNames.push_back(m_inputNamesPtrs[i].get());
        Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputDims = inputTensorInfo.GetShape();
        m_inputNodeDims.push_back(inputDims);
    }
    for (size_t i = 0; i < numOutputNodes; i++) {
        m_outputNamesPtrs.push_back(std::move(m_session->GetOutputNameAllocated(i, allocator)));
        m_outputNames.push_back(m_outputNamesPtrs[i].get());
        Ort::TypeInfo outputTypeInfo = m_session->GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();
        m_outputNodeDims.push_back(outputDims);
    }
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];

    std::string warpTempName = m_modelsJson->at("faceEnhancerModels").at(m_modelName).at("template");
    auto fVec = m_modelsJson->at("faceHelper").at("warpTemplate").at(warpTempName).get<std::vector<float>>();
    m_warpTemplate.clear();
    for (int i = 0; i < fVec.size(); i += 2) {
        m_warpTemplate.emplace_back(fVec.at(i), fVec.at(i + 1));
    }
    auto iVec = m_modelsJson->at("faceEnhancerModels").at(m_modelName).at("size").get<std::vector<int>>();
    m_size = cv::Size(iVec.at(0), iVec.at(1));
}

void FaceEnhancer::processImage(const std::string &targetPath, const std::string &outputPath) {
    Typing::Faces referenceFaces{};
    Typing::VisionFrame targetFrame = Vision::readStaticImage(targetPath);
    auto result = processFrame(referenceFaces, targetFrame);
    if (result) {
        Vision::writeImage(*result, outputPath);
    }
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::processFrame(const Typing::Faces &referenceFaces,
                           const Typing::VisionFrame &targetFrame) {
    if (m_faceAnalyser == nullptr) {
        throw std::runtime_error("Face analyser is not set.");
    }

    std::shared_ptr<Typing::VisionFrame> resultFrame = std::make_shared<Typing::VisionFrame>(targetFrame);
    if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_Many) {
        auto manyTargetFaces = m_faceAnalyser->getManyFaces(targetFrame);
        if (!manyTargetFaces->empty()) {
            for (auto &targetFace : *manyTargetFaces) {
                resultFrame = enhanceFace(targetFace, *resultFrame);
            }
        }
    } else if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_One) {
        // Todo
    } else if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_Reference) {
        // Todo
    }

    return resultFrame;
}

void FaceEnhancer::setFaceAnalyser(const std::shared_ptr<FaceAnalyser> &faceAnalyser) {
    m_faceAnalyser = faceAnalyser;
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::enhanceFace(const Face &targetFace, const VisionFrame &tempVisionFrame) {
    init();

    // Todo
    auto cropVisionAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(tempVisionFrame,
                                                                       targetFace.faceLandMark5_68,
                                                                       m_warpTemplate, m_size);
    auto cropBoxMask = FaceMasker::createStaticBoxMask(std::get<0>(*cropVisionAndAffineMat).size(),
                                                       Globals::faceMaskBlur, Globals::faceMaskPadding);
    std::list<cv::Mat> cropMaskList{*cropBoxMask};

    std::vector<cv::Mat> bgrChannels(3);
    split(std::get<0>(*cropVisionAndAffineMat), bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0 * 0.5), -1.0);
    }
    const int imageArea = m_inputHeight * this->m_inputWidth;
    m_inputImageData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(this->m_inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize); /// rgb顺序
    memcpy(this->m_inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(this->m_inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);

    std::vector<Ort::Value> inputTensors;
    for (const auto &name : m_inputNames) {
        std::string strName(name);
        if (strName == "input") {
            std::vector<int64_t> inputImgShape = {1, 3, m_inputHeight, m_inputWidth};
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(m_memoryInfo, m_inputImageData.data(),
                                                                     m_inputImageData.size(),
                                                                     inputImgShape.data(),
                                                                     inputImgShape.size());
            inputTensors.push_back(std::move(inputTensor));
        } else if (strName == "weight") {
            std::vector<int64_t> weightsShape = {1, 1};
            std::vector<double> weightsData = {1.0};
            Ort::Value weightsTensor = Ort::Value::CreateTensor<double>(m_memoryInfo, weightsData.data(),
                                                                      weightsData.size(),
                                                                      weightsShape.data(),
                                                                      weightsShape.size());
            inputTensors.push_back(std::move(weightsTensor));
        }
    }

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensor = m_session->Run(runOptions, m_inputNames.data(),
                                                          inputTensors.data(), inputTensors.size(),
                                                          m_outputNames.data(), m_outputNames.size());

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
        mat.setTo(-1, mat < -1);
        mat.setTo(1, mat > 1);
        mat = (mat + 1) * 0.5;
        mat *= 255.f;
        mat.setTo(0, mat < 0);
        mat.setTo(255, mat > 255);
    }
    // Merge the channels into a single matrix
    cv::Mat resultMat;
    cv::merge(channelMats, resultMat);
    resultMat.convertTo(resultMat, CV_8UC3);

    for (auto &cropMask : cropMaskList) {
        cropMask.setTo(0, cropMask < 0);
        cropMask.setTo(1, cropMask > 1);
    }

    cv::Mat dstImage = FaceHelper::pasteBack(tempVisionFrame, resultMat, cropMaskList.front(),
                                             std::get<1>(*cropVisionAndAffineMat));

    return std::make_shared<Typing::VisionFrame>(std::move(dstImage));
}

} // namespace Ffc
