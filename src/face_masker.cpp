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
                       const std::shared_ptr<nlohmann::json> &modelsInfoJson,
                       const std::shared_ptr<const Config> &config) :
    m_modelsInfoJson(modelsInfoJson), m_config(config) {
    m_env = env;
    m_maskerMap = std::make_shared<std::unordered_map<Method, std::shared_ptr<OrtSession>>>();
}

std::shared_ptr<cv::Mat> FaceMasker::createOcclusionMask(const Typing::VisionFrame &cropVisionFrame) {
    std::string modelPath = m_modelsInfoJson->at("faceMaskerModels").at("face_occluder").at("path");
    // 检查model是否存在，若不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceMaskerModels").at("face_occluder").at("url"),
                                                    "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }

    std::shared_ptr<OrtSession> session = nullptr;
    if (!m_maskerMap->contains(Method::Occlusion)) {
        session = std::make_shared<OrtSession>(m_env);
        m_maskerMap->emplace(Method::Occlusion, session);
    } else {
        session = m_maskerMap->at(Method::Occlusion);
    }
    session->createSession(modelPath);
    const int inputHeight = session->m_inputNodeDims[0][1];
    const int inputWidth = session->m_inputNodeDims[0][2];

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
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(session->m_memoryInfo, inputImageData.data(),
                                                              inputImageData.size(),
                                                              inputImageShape.data(),
                                                              inputImageShape.size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensors = session->m_session->Run(runOptions, session->m_inputNames.data(),
                                                                    inputTensors.data(), inputTensors.size(),
                                                                    session->m_outputNames.data(), session->m_outputNames.size());

    float *pdata = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outsShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int outputHeight = outsShape[1];
    const int outputWidth = outsShape[2];

    cv::Mat mask(outputHeight, outputWidth, CV_32FC1, pdata);
    mask.setTo(0, mask < 0); // 其实没必要，但与python版本保持一致
    mask.setTo(1, mask > 1);

    cv::resize(mask, mask, cropVisionFrame.size());

    cv::GaussianBlur(mask, mask, cv::Size(0, 0), 5);
    mask.setTo(0, mask < 0); // 其实没必要，但与python版本保持一致
    mask.setTo(1, mask > 1);

    mask.setTo(0.5, mask < 0.5);
    mask.setTo(1, mask > 1);
    mask = (mask - 0.5) * 2;
    return std::make_shared<cv::Mat>(std::move(mask));
}

std::shared_ptr<cv::Mat> FaceMasker::createRegionMask(const Typing::VisionFrame &cropVisionFrame,
                                                      const std::unordered_set<Typing::EnumFaceMaskRegion> &regions) {
    std::string modelPath = m_modelsInfoJson->at("faceMaskerModels").at("face_parser").at("path");
    // 检查model是否存在，若不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceMaskerModels").at("face_parser").at("url"),
                                                    "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }

    std::shared_ptr<OrtSession> session = nullptr;
    if (!m_maskerMap->contains(Method::Region)) {
        session = std::make_shared<OrtSession>(m_env);
        session->createSession(modelPath);
        m_maskerMap->emplace(Method::Region, session);
    } else {
        session = m_maskerMap->at(Method::Region);
    }
    const int inputHeight = session->m_inputNodeDims[0][2];
    const int inputWidth = session->m_inputNodeDims[0][3];

    cv::Mat inputImage = cropVisionFrame.clone();
    cv::resize(cropVisionFrame, inputImage, cv::Size(inputHeight, inputWidth));
    cv::flip(inputImage, inputImage, 1);
    std::vector<cv::Mat> bgrChannels;
    cv::split(inputImage, bgrChannels);
    for (int i = 0; i < 3; i++) {
        bgrChannels[i].convertTo(bgrChannels[i], CV_32FC1, 1.0 / 127.5, -1.0);
    }
    const int imageArea = inputHeight * inputWidth;
    std::vector<float> inputImageData(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize); /// rgb顺序
    memcpy(inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);

    std::vector<int64_t> inputImageShape{1, 3, inputHeight, inputWidth};
    std::vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(session->m_memoryInfo, inputImageData.data(),
                                                              inputImageData.size(),
                                                              inputImageShape.data(),
                                                              inputImageShape.size()));

    Ort::RunOptions runOptions;
    std::vector<Ort::Value> outputTensors = session->m_session->Run(runOptions, session->m_inputNames.data(),
                                                                    inputTensors.data(), inputTensors.size(),
                                                                    session->m_outputNames.data(), session->m_outputNames.size());

    float *pdata = outputTensors[0].GetTensorMutableData<float>();
    std::vector<int64_t> outsShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const int outputHeight = outsShape[2];
    const int outputWidth = outsShape[3];
    const int outputArea = outputHeight * outputWidth;

    std::vector<cv::Mat> masks;
    if (regions.contains(Typing::EnumFaceMaskRegion::All)) {
        for (const auto &region : Typing::faceMaskRegionAllSet) {
            int regionInt = static_cast<int>(region);
            cv::Mat regionMask(outputHeight, outputWidth, CV_32FC1, pdata + regionInt * outputArea);
            masks.emplace_back(std::move(regionMask));
        }
    } else {
        for (const auto &region : regions) {
            int regionInt = static_cast<int>(region);
            cv::Mat regionMask(outputHeight, outputWidth, CV_32FC1, pdata + regionInt * outputArea);
            regionMask.setTo(0, regionMask < 0);
            regionMask.setTo(1, regionMask > 1);
            masks.emplace_back(std::move(regionMask));
        }
    }

    cv::Mat resultMask = masks.front().clone();
    for (size_t i = 1; i < masks.size(); ++i) {
        resultMask = cv::max(resultMask, masks[i]);
    }
    resultMask.setTo(0, resultMask < 0);
    resultMask.setTo(1, resultMask > 1);

    cv::resize(resultMask, resultMask, cropVisionFrame.size());
    resultMask.setTo(0, resultMask < 0);
    resultMask.setTo(1, resultMask > 1);

    cv::GaussianBlur(resultMask, resultMask, cv::Size(0, 0), 5);
    resultMask.setTo(0.5, resultMask < 0.5);
    resultMask.setTo(1, resultMask > 1);

    resultMask = (resultMask - 0.5) * 2;
    return std::make_shared<cv::Mat>(std::move(resultMask));
}

std::shared_ptr<cv::Mat> FaceMasker::getBestMask(const std::vector<cv::Mat> &masks) {
    if (masks.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }

    // Initialize the minMask with the first mask in the vector
    cv::Mat minMask = masks[0].clone();

    for (size_t i = 1; i < masks.size(); ++i) {
        // Check if the current mask has the same size and type as minMask
        if (masks[i].size() != minMask.size() || masks[i].type() != minMask.type()) {
            m_logger->error("[FaceMasker] All masks must have the same size and type.");
        }
        cv::min(minMask, masks[i], minMask);
    }
    minMask.setTo(0, minMask < 0);
    minMask.setTo(1, minMask > 1);

    return std::make_shared<cv::Mat>(std::move(minMask));
}

bool FaceMasker::preCheck() {
    std::string downloadDir = FileSystem::resolveRelativePath("./models");
    std::vector<std::string> modelUrls = {
        m_modelsInfoJson->at("faceMaskerModels").at("face_occluder").at("url"),
        m_modelsInfoJson->at("faceMaskerModels").at("face_parser").at("url")};
    std::vector<std::string> modelPaths = {
        m_modelsInfoJson->at("faceMaskerModels").at("face_occluder").at("path"),
        m_modelsInfoJson->at("faceMaskerModels").at("face_parser").at("path")};

    for (size_t i = 0; i < modelUrls.size(); ++i) {
        modelPaths.at(i) = FileSystem::resolveRelativePath(modelPaths.at(i));
        if (!FileSystem::isFile(modelPaths.at(i))) {
            if (!m_config->m_skipDownload) {
                bool downloadSuccess = Downloader::download(modelUrls.at(i), downloadDir);
                if (!downloadSuccess) {
                    m_logger->error(std::format("[FaceMasker] Failed to download the model file: {}", modelUrls.at(i)));
                    return false;
                }
            } else {
                m_logger->error(std::format("[FaceMasker] Model file not found: {}", modelPaths.at(i)));
                return false;
            }
        }
    }
    return true;
}
} // namespace Ffc