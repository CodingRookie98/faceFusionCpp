/**
 ******************************************************************************
 * @file           : face_swapper.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#include "face_swapper.h"

namespace Ffc {
FaceSwapper::FaceSwapper(const std::shared_ptr<Ort::Env> &env,
                         const std::shared_ptr<FaceAnalyser> &faceAnalyser,
                         const std::shared_ptr<FaceMasker> &faceMasker,
                         const std::shared_ptr<nlohmann::json> &modelsInfoJson,
                         const std::shared_ptr<const Config> &config) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson), m_config(config) {
    m_faceAnalyser = faceAnalyser;
    m_faceMasker = faceMasker;
}

void FaceSwapper::processImage(const std::unordered_set<std::string> &sourcePaths,
                               const std::string &targetPath,
                               const std::string &outputPath) {
    Typing::Faces referenceFaces;
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference) {
        auto referenceFacesMap = m_faceStore->getReferenceFaces();
        for (const auto &referenceFace : referenceFacesMap) {
            for (const auto &face : referenceFace.second) {
                referenceFaces.push_back(face);
            }
        }
        if (referenceFaces.empty()) {
            m_logger->error("[FaceSwapper] You must provide at least one reference face.");
            std::exit(1);
        }
    }

    std::vector<cv::Mat> sourceFrames = Ffc::Vision::readStaticImages(sourcePaths);
    std::shared_ptr<Typing::Face> sourceFace = m_faceAnalyser->getAverageFace(sourceFrames);
    auto targetFrame = Ffc::Vision::readStaticImage(targetPath);

    auto resultFrame = processFrame(referenceFaces, *sourceFace, targetFrame);
    if (resultFrame == nullptr) {
        m_logger->error("[FaceSwapper] Failed to process image");
        return;
    }
    Ffc::Vision::writeImage(*resultFrame, outputPath);
}

void FaceSwapper::processImages(const std::unordered_set<std::string> &sourcePaths,
                                const std::vector<std::string> &targetPaths,
                                const std::vector<std::string> &outputPaths) {
    if (targetPaths.size() != outputPaths.size()) {
        m_logger->error("[FaceSwapper] The number of target paths and output paths must be equal!");
        throw std::invalid_argument("[FaceSwapper] The number of target paths and output paths must be equal");
    }
    if (sourcePaths.empty()) {
        m_logger->error("[FaceSwapper] No source paths were provided, so face swapping was skipped.");
        return;
    }
    if (targetPaths.empty()) {
        m_logger->error("[FaceSwapper] No target paths provided!");
        return;
    }

    Typing::Faces referenceFaces;
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference) {
        auto referenceFacesMap = m_faceStore->getReferenceFaces();
        for (const auto &referenceFace : referenceFacesMap) {
            for (const auto &face : referenceFace.second) {
                referenceFaces.push_back(face);
            }
        }
        if (referenceFaces.empty()) {
            m_logger->error("[FaceSwapper] You must provide at least one reference face.");
            std::exit(1);
        }
    }

    std::vector<cv::Mat> sourceFrames = Ffc::Vision::readStaticImages(sourcePaths);
    std::shared_ptr<Typing::Face> sourceFace = m_faceAnalyser->getAverageFace(sourceFrames);
    dp::thread_pool poolForProcess(m_config->m_executionThreadCount);
    dp::thread_pool poolForWriteImage(std::thread::hardware_concurrency());
    std::vector<std::future<bool>> writeImageResults;
    for (int i = 0; i < targetPaths.size(); ++i) {
        auto targetFrame = Ffc::Vision::readStaticImage(targetPaths[i]);
        poolForProcess.enqueue([referenceFaces, sourceFace, targetFrame, &poolForWriteImage,
                                &writeImageResults, outputPath = outputPaths[i], this]() -> void {
            auto frame = processFrame(referenceFaces, *sourceFace, targetFrame);
            static std::mutex writeImageMutex;
            std::lock_guard<std::mutex> lock(writeImageMutex);
            writeImageResults.emplace_back(poolForWriteImage.enqueue([writeFrame = std::move(frame), outputPath]() {
                return Ffc::Vision::writeImage(*writeFrame, outputPath);
            }));
        });
    }

    const size_t numTargetPaths = targetPaths.size();
    ProgressBar bar;
    show_console_cursor(false);
    bar.setMaxProgress(100);
    bar.setPrefixText("[FaceSwapper] Processing images");
    bar.setPostfixText(std::format("{}/{}", 0, targetPaths.size()));
    bar.setProgress(0);
    int i = 0;
    static bool isAllWriteSuccess = true;
    while (true) {
        if (writeImageResults.size() <= i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        } else {
            if (!writeImageResults[i].valid()) {
                isAllWriteSuccess = false;
                m_logger->error(std::format("[FaceSwapper] Failed to process image: {}", targetPaths[i]));
                ++i;
                continue;
            }
            
            auto writeIsSuccess = writeImageResults[i].get();
            if (!writeIsSuccess) {
                isAllWriteSuccess = false;
                m_logger->error(std::format("[FaceSwapper] Failed to write image: {}", outputPaths[i]));
            }

            bar.setPostfixText(std::format("{}/{}", (i + 1), numTargetPaths));
            int progress = static_cast<int>(std::round(((i + 1) * 100.0f) / numTargetPaths));
            bar.setProgress(progress);
            
            ++i;
            if (i >= outputPaths.size()) {
                break;
            }
        }
    }
    show_console_cursor(true);
    if(!isAllWriteSuccess) {
        m_logger->error("[FaceSwapper] Some images failed to process or write.");
    }
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::processFrame(const Typing::Faces &referenceFaces,
                                                               const Face &sourceFace,
                                                               const VisionFrame &targetFrame) {
    if (m_faceAnalyser == nullptr) {
        m_logger->error("[FaceSwapper] Face analyser is not set");
        throw std::runtime_error("[FaceSwapper] Face analyser is not set");
    }

    std::shared_ptr<Typing::VisionFrame> resultFrame = std::make_shared<Typing::VisionFrame>(targetFrame);
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Many) {
        auto manyTargetFaces = m_faceAnalyser->getManyFaces(targetFrame);
        if (manyTargetFaces != nullptr && !manyTargetFaces->empty()) {
            for (auto &targetFace : *manyTargetFaces) {
                resultFrame = swapFace(sourceFace, targetFace, *resultFrame);
            }
        }
    } else if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_One) {
        auto targetFace = m_faceAnalyser->getOneFace(targetFrame);
        if (targetFace != nullptr) {
            resultFrame = swapFace(sourceFace, *targetFace, *resultFrame);
        }
    } else if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference) {
        Typing::Faces similarFaces = m_faceAnalyser->findSimilarFaces(referenceFaces, targetFrame, m_config->m_referenceFaceDistance);
        if (!similarFaces.empty()) {
            for (const auto &similarFace : similarFaces) {
                resultFrame = swapFace(sourceFace, similarFace, *resultFrame);
            }
        } else {
            m_logger->warn("[FaceSwapper] No similar faces found, so skip this image process!");
        }
    }

    return resultFrame;
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::swapFace(const Face &sourceFace,
                                                           const Face &targetFace,
                                                           const VisionFrame &targetFrame) {
    auto resultFrame = this->applySwap(sourceFace, targetFace, targetFrame);
    return resultFrame;
}

void FaceSwapper::init() {
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];

    auto templateName = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("template");
    auto fVec = m_modelsInfoJson->at("faceHelper").at("warpTemplate").at(templateName).get<std::vector<float>>();
    for (int i = 0; i < fVec.size(); i += 2) {
        m_warpTemplate.emplace_back(fVec.at(i), fVec.at(i + 1));
    }

    m_mean = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("mean").get<std::vector<float>>();

    m_standardDeviation = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("standard_deviation").get<std::vector<float>>();
    auto sizeVec = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("size").get<std::vector<int>>();
    m_size = cv::Size(sizeVec.at(0), sizeVec.at(1));
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::applySwap(const Face &sourceFace, const Face &targetFace, const VisionFrame &targetFrame) {
    auto croppedTargetFrameAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(
        targetFrame, targetFace.faceLandMark5_68, m_warpTemplate, m_size);
    auto preparedTargetFrameBGR = Ffc::FaceSwapper::prepareCropVisionFrame(std::get<0>(*croppedTargetFrameAndAffineMat), m_mean, m_standardDeviation);

    std::vector<cv::Mat> cropMasks;
    if (m_config->m_faceMaskTypeSet.contains(Typing::EnumFaceMaskerType::FM_Box)) {
        auto boxMask = FaceMasker::createStaticBoxMask(std::get<0>(*croppedTargetFrameAndAffineMat).size(),
                                                       m_config->m_faceMaskBlur, m_config->m_faceMaskPadding);
        cropMasks.push_back(std::move(*boxMask));
    } else if (m_config->m_faceMaskTypeSet.contains(Typing::EnumFaceMaskerType::FM_Occlusion)) {
        auto occlusionMask = m_faceMasker->createOcclusionMask(std::get<0>(*croppedTargetFrameAndAffineMat));
        cropMasks.push_back(std::move(*occlusionMask));
    }

    // Create input tensors
    std::vector<Ort::Value> inputTensors;
    std::string modelType = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("type").get<std::string>();
    std::shared_ptr<std::vector<float>> inputImageData;
    for (const auto &inputName : m_inputNames) {
        if (std::string(inputName) == "source") {
            if (modelType == "blendswap" || modelType == "uniface") {
                auto preparedSourceFrame = this->prepareSourceFrame(sourceFace);
                inputImageData = this->prepareCropFrameData(*preparedSourceFrame);
                std::vector<int64_t> inputImageShape = {1, 3, preparedSourceFrame->rows, preparedSourceFrame->cols};
                inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, inputImageData->data(), inputImageData->size(), inputImageShape.data(), inputImageShape.size()));
            } else {
                static auto inputEmbeddingData = this->prepareSourceEmbedding(sourceFace);
                std::vector<int64_t> inputEmbeddingShape = {1, (int64_t)inputEmbeddingData->size()};
                inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, inputEmbeddingData->data(), inputEmbeddingData->size(), inputEmbeddingShape.data(), inputEmbeddingShape.size()));
            }
        } else if (std::string(inputName) == "target") {
            inputImageData = this->prepareCropFrameData(*preparedTargetFrameBGR);
            std::vector<int64_t> inputImageShape = {1, 3, m_inputHeight, m_inputWidth};
            inputTensors.emplace_back(Ort::Value::CreateTensor<float>(m_memoryInfo, inputImageData->data(), inputImageData->size(), inputImageShape.data(), inputImageShape.size()));
        }
    }

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

    if (m_config->m_faceMaskTypeSet.contains(Typing::EnumFaceMaskerType::FM_Region)) {
        auto regionMask = m_faceMasker->createRegionMask(resultMat, m_config->m_faceMaskRegionsSet);
        cropMasks.push_back(std::move(*regionMask));
    }
    for (auto &cropMask : cropMasks) {
        cropMask.setTo(0, cropMask < 0);
        cropMask.setTo(1, cropMask > 1);
    }

    auto bestCropMask = m_faceMasker->getBestMask(cropMasks);

    auto dstImage = FaceHelper::pasteBack(targetFrame, resultMat, *bestCropMask,
                                          std::get<1>(*croppedTargetFrameAndAffineMat));

    return dstImage;
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::prepareCropVisionFrame(const VisionFrame &visionFrame, const std::vector<float> &mean, const std::vector<float> &standDeviation) {
    cv::Mat bgrImage = visionFrame.clone();
    std::vector<cv::Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels.at(c), CV_32FC1, 1 / (255.0 * standDeviation.at(c)),
                                 -mean.at(c) / (float)standDeviation.at(c));
    }

    cv::Mat processedBGR;
    cv::merge(bgrChannels, processedBGR);

    return std::make_shared<Typing::VisionFrame>(processedBGR);
}

std::shared_ptr<std::vector<float>> FaceSwapper::prepareSourceEmbedding(const Face &sourceFace) {
    std::vector<float> result;
    if (m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("type") == "inswapper") {
        double norm = cv::norm(sourceFace.embedding, cv::NORM_L2);
        int lenFeature = sourceFace.embedding.size();
        result.resize(lenFeature);
        for (int i = 0; i < lenFeature; ++i) {
            double sum = 0.0f;
            for (int j = 0; j < lenFeature; ++j) {
                sum += sourceFace.embedding.at(j)
                       * m_initializerArray.at(j * lenFeature + i);
            }
            result.at(i) = (float)(sum / norm);
        }
    } else {
        result = sourceFace.normedEmbedding;
    }
    return std::make_shared<std::vector<float>>(std::move(result));
}

std::shared_ptr<std::vector<float>> FaceSwapper::prepareCropFrameData(const VisionFrame &cropFrame) const {
    cv::Mat bgrImage = cropFrame.clone();
    std::vector<cv::Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);
    const int imageArea = bgrImage.rows * bgrImage.cols;
    std::vector<float> inputImageData(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize);                 // R
    memcpy(inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);     // G
    memcpy(inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize); // B
    return std::make_shared<std::vector<float>>(std::move(inputImageData));
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::prepareSourceFrame(const Face &sourceFace) const {
    std::string modelType = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("type").get<std::string>();
    Typing::VisionFrame sourceVisionFrame = Vision::readStaticImage(*m_config->m_sourcePaths.begin());
    Typing::VisionFrame croppedVisionFrame;
    if (modelType == "blendswap") {
        auto cropFrameAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(sourceVisionFrame,
                                                                          sourceFace.faceLandMark5_68,
                                                                          m_warpTemplate,
                                                                          cv::Size(112, 112));
        croppedVisionFrame = std::get<0>(*cropFrameAndAffineMat);
    }
    if (modelType == "uniface") {
        auto cropFrameAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(sourceVisionFrame, sourceFace.faceLandMark5_68,
                                                                          m_warpTemplate,
                                                                          m_size);
        croppedVisionFrame = std::get<0>(*cropFrameAndAffineMat);
    }

    std::vector<cv::Mat> bgrChannels(3);
    split(croppedVisionFrame, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels.at(c), CV_32FC1, 1 / 255.0);
    }

    cv::Mat processedBGR;
    cv::merge(bgrChannels, processedBGR);
    return std::make_shared<Typing::VisionFrame>(std::move(processedBGR));
}

bool FaceSwapper::preCheck() {
    m_logger->info("[FaceSwapper] pre check");
    switch (m_config->m_faceSwapperModel) {
    case Typing::FSM_Inswapper_128:
        m_modelName = "inswapper_128";
        break;
    case Typing::FSM_Inswapper_128_fp16:
        m_modelName = "inswapper_128_fp16";
        break;
    case Typing::FSM_Blendswap_256:
        m_modelName = "blendswap_256";
        break;
    case Typing::FSM_Simswap_256:
        m_modelName = "simswap_256";
        break;
    case Typing::FSM_Simswap_512_unofficial:
        m_modelName = "simswap_512_unofficial";
        break;
    case Typing::FSM_Uniface_256:
        m_modelName = "uniface_256";
        break;
    default:
        break;
    }

    if (m_modelName.empty()) {
        m_logger->error("Invalid face swapper model, supported models: blendswap_256 inswapper_128 inswapper_128_fp16 simswap_256 simswap_512_unofficial uniface_256");
        return false;
    }

    std::string modelPath = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("path");
    std::string modelUrl = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("url");
    std::string downloadDir = FileSystem::resolveRelativePath("./models");

    // 检查modelPath是否存在, 不存在则下载
    if (!FileSystem::isFile(modelPath)) {
        if (!m_config->m_skipDownload) {
            bool downloadSuccess = Downloader::download(modelUrl, downloadDir);
            if (!downloadSuccess) {
                m_logger->error(std::format("[FaceSwapper] Failed to download the model file: {}", modelUrl));
                return false;
            }
        } else {
            m_logger->error(std::format("[FaceSwapper] Model file is not Found: {}", modelPath));
            return false;
        }
    }
    return true;
}

bool FaceSwapper::postCheck() {
    m_logger->info("[FaceSwapper] post check");
    std::string modelUrl = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("url");
    std::string modelPath = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("path");
    modelPath = FileSystem::resolveRelativePath(modelPath);

    if (!m_config->m_skipDownload && !Downloader::isDownloadDone(modelUrl, modelPath)) {
        m_logger->error("[FaceSwapper] Model file is not downloaded: " + Downloader::getFileNameFromUrl(modelUrl));
        return false;
    }
    if (!FileSystem::isFile(modelPath)) {
        m_logger->error("[FaceSwapper] Model file is not present: " + modelPath);
        return false;
    }
    return true;
}

bool FaceSwapper::preProcess() {
    m_logger->info("[FaceSwapper] pre process");

    if (m_faceSwapperModel == nullptr || *m_faceSwapperModel != m_config->m_faceSwapperModel) {
        if (m_faceSwapperModel != nullptr && *m_faceSwapperModel != m_config->m_faceSwapperModel) {
            postCheck();
        }
        m_faceSwapperModel = std::make_shared<Typing::EnumFaceSwapperModel>(m_config->m_faceSwapperModel);

        std::string modelPath = m_modelsInfoJson->at("faceSwapperModels").at(m_modelName).at("path");

        if (m_modelName == "inswapper_128" || m_modelName == "inswapper_128_fp16") {
            // Load ONNX model as a protobuf message
            onnx::ModelProto modelProto;
            std::string model128Path = m_modelsInfoJson->at("faceSwapperModels").at("inswapper_128").at("path");
            std::ifstream input(modelPath, std::ios::binary);
            if (!modelProto.ParseFromIstream(&input)) {
                m_logger->error("[FaceSwapper] Failed to parse the model: " + m_modelName);
                throw std::runtime_error("Failed to load model.");
            }
            // Access the initializer
            const onnx::TensorProto &initializer = modelProto.graph().initializer(modelProto.graph().initializer_size() - 1);
            // Convert initializer to an array
            m_initializerArray.clear();
            if (m_modelName == "inswapper_128") {
                m_initializerArray.assign(initializer.float_data().begin(), initializer.float_data().end());
            } else {
                // fp_16
                std::string rawData = initializer.raw_data();
                auto data = reinterpret_cast<const float *>(rawData.data());
                m_initializerArray.assign(data, data + rawData.size() / sizeof(float));
            }
        } else {
            m_initializerArray.clear();
        }

        this->createSession(modelPath);
        init();
    }

    return true;
}

Typing::VisionFrame FaceSwapper::getReferenceFrame(const Face &sourceFace, const Face &targetFace, const VisionFrame &tempVisionFrame) {
    return *swapFace(sourceFace, targetFace, tempVisionFrame);
}

} // namespace Ffc
