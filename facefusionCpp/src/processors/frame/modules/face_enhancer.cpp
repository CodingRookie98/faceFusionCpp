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

namespace Ffc {
FaceEnhancer::FaceEnhancer(const std::shared_ptr<Ort::Env> &env,
                           const std::shared_ptr<FaceAnalyser> &faceAnalyser,
                           const std::shared_ptr<FaceMasker> &faceMasker,
                           const std::shared_ptr<nlohmann::json> &modelsInfoJson,
                           const std::shared_ptr<const Config> &config) :
    OrtSession(env), m_modelsInfoJson(modelsInfoJson), m_config(config) {
    m_faceAnalyser = faceAnalyser;
    m_faceMasker = faceMasker;
}

void FaceEnhancer::init() {
    std::string modelPath = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("path");

    // 检查modelPath文件是否存在，不存在则下载
    if (!FileSystem::fileExists(modelPath)) {
        bool downloadSuccess = Downloader::download(m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("url"),
                                                    "./models");
        if (!downloadSuccess) {
            throw std::runtime_error("Failed to download the model file: " + modelPath);
        }
    }
    m_inputHeight = (int)m_inputNodeDims[0][2];
    m_inputWidth = (int)m_inputNodeDims[0][3];

    std::string warpTempName = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("template");
    auto fVec = m_modelsInfoJson->at("faceHelper").at("warpTemplate").at(warpTempName).get<std::vector<float>>();
    m_warpTemplate.clear();
    for (int i = 0; i < fVec.size(); i += 2) {
        m_warpTemplate.emplace_back(fVec.at(i), fVec.at(i + 1));
    }

    auto iVec = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("size").get<std::vector<int>>();
    m_size = cv::Size(iVec.at(0), iVec.at(1));
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::processFrame(const Typing::Faces &referenceFaces,
                           const Typing::VisionFrame &targetFrame) {
    std::shared_ptr<Typing::VisionFrame> resultFrame = std::make_shared<Typing::VisionFrame>(targetFrame);
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Many) {
        auto manyTargetFaces = m_faceAnalyser->getManyFaces(targetFrame);
        if (manyTargetFaces != nullptr && !manyTargetFaces->empty()) {
            for (auto &targetFace : *manyTargetFaces) {
                resultFrame = enhanceFace(targetFace, *resultFrame);
            }
        }
    } else if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_One) {
        auto targetFace = m_faceAnalyser->getOneFace(targetFrame);
        if (targetFace != nullptr) {
            resultFrame = enhanceFace(*targetFace, *resultFrame);
        }
    } else if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference) {
        Typing::Faces similarFaces = m_faceAnalyser->findSimilarFaces(referenceFaces, targetFrame, m_config->m_referenceFaceDistance);
        if (!similarFaces.empty()) {
            for (const auto &similarFace : similarFaces) {
                resultFrame = enhanceFace(similarFace, *resultFrame);
            }
        } else {
            m_logger->error("[FaceEnhancer] No similar faces found");
        }
    }

    return resultFrame;
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::enhanceFace(const Face &targetFace, const VisionFrame &tempVisionFrame) {
    auto result = applyEnhance(targetFace, tempVisionFrame);
    return result;
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::blendFrame(const VisionFrame &targetFrame,
                         const VisionFrame &pasteVisionFrame) {
    const float faceEnhancerBlend = 1 - ((float)m_config->m_faceEnhancerBlend / 100.f);
    cv::Mat dstImage;
    cv::addWeighted(targetFrame, faceEnhancerBlend, pasteVisionFrame, 1 - faceEnhancerBlend, 0, dstImage);
    return std::make_shared<Typing::VisionFrame>(std::move(dstImage));
}

std::shared_ptr<Typing::VisionFrame>
FaceEnhancer::applyEnhance(const Face &targetFace, const VisionFrame &tempVisionFrame) {
    auto cropVisionAndAffineMat = FaceHelper::warpFaceByFaceLandmarks5(tempVisionFrame,
                                                                       targetFace.faceLandMark5_68,
                                                                       m_warpTemplate, m_size);
    std::vector<cv::Mat> cropMasks;
    auto cropBoxMask = FaceMasker::createStaticBoxMask(std::get<0>(*cropVisionAndAffineMat).size(),
                                                       m_config->m_faceMaskBlur, m_config->m_faceMaskPadding);
    cropMasks.emplace_back(*cropBoxMask);
    
    if (m_config->m_faceMaskTypeSet.contains(Typing::EnumFaceMaskerType::FM_Occlusion)) {
        auto cropOcclusionMask = m_faceMasker->createOcclusionMask(std::get<0>(*cropVisionAndAffineMat));
        cropMasks.emplace_back(*cropOcclusionMask);
    }

    std::vector<cv::Mat> bgrChannels(3);
    split(std::get<0>(*cropVisionAndAffineMat), bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / (255.0 * 0.5), -1.0);
    }

    const int imageArea = m_inputHeight * m_inputWidth;
    std::vector<float> inputImageData;
    inputImageData.resize(3 * imageArea);
    size_t singleChnSize = imageArea * sizeof(float);
    memcpy(inputImageData.data(), (float *)bgrChannels[2].data, singleChnSize); /// rgb顺序
    memcpy(inputImageData.data() + imageArea, (float *)bgrChannels[1].data, singleChnSize);
    memcpy(inputImageData.data() + imageArea * 2, (float *)bgrChannels[0].data, singleChnSize);

    std::vector<Ort::Value> inputTensors;
    for (const auto &name : m_inputNames) {
        std::string strName(name);
        if (strName == "input") {
            std::vector<int64_t> inputImgShape = {1, 3, m_inputHeight, m_inputWidth};
            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(m_memoryInfo, inputImageData.data(),
                                                                     inputImageData.size(),
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

    for (auto &cropMask : cropMasks) {
        cropMask.setTo(0, cropMask < 0);
        cropMask.setTo(1, cropMask > 1);
    }
    auto bestMask = m_faceMasker->getBestMask(cropMasks);

    auto dstImage = FaceHelper::pasteBack(tempVisionFrame, resultMat, *bestMask,
                                          std::get<1>(*cropVisionAndAffineMat));
    dstImage = blendFrame(tempVisionFrame, *dstImage);

    return dstImage;
}

bool FaceEnhancer::postCheck() {
    m_logger->info("[FaceEnhancer] post check");
    std::string modelUrl = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("url");
    std::string modelPath = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("path");
    modelPath = FileSystem::resolveRelativePath(modelPath);

    if (!m_config->m_skipDownload && !Downloader::isDownloadDone(modelUrl, modelPath)) {
        m_logger->error("[FaceEnhancer] Model file is not downloaded: " + Downloader::getFileNameFromUrl(modelUrl));
        return false;
    }
    if (!FileSystem::isFile(modelPath)) {
        m_logger->error("[FaceEnhancer] Model file is not present: " + modelPath);
        return false;
    }
    return true;
}

bool FaceEnhancer::preCheck() {
    m_logger->info("[FaceEnhancer] pre check");

    switch (m_config->m_faceEnhancerModel) {
    case Typing::FEM_Gfpgan_12:
        m_modelName = "gfpgan_1.2";
        break;
    case Typing::FEM_Gfpgan_13:
        m_modelName = "gfpgan_1.3";
        break;
    case Typing::FEM_Gfpgan_14:
        m_modelName = "gfpgan_1.4";
        break;
    case Typing::FEM_CodeFormer:
        m_modelName = "codeformer";
        break;
    case Typing::FEM_Gpen_bfr_256:
        m_modelName = "gpen_bfr_256";
        break;
    case Typing::FEM_Gpen_bfr_512:
        m_modelName = "gpen_bfr_512";
        break;
    case Typing::FEM_Gpen_bfr_1024:
        m_modelName = "gpen_bfr_1024";
        break;
    case Typing::FEM_Gpen_bfr_2048:
        m_modelName = "gpen_bfr_2048";
        break;
    case Typing::FEM_Restoreformer_plus_plus:
        m_modelName = "restoreformer_plus_plus";
        break;
    default:
        m_modelName = "gfpgan_1.4";
        break;
    }
    std::string modelPath = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("path");
    std::string modelUrl = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("url");
    std::string downloadDir = FileSystem::resolveRelativePath("./models");

    // 检查modelPath是否存在, 不存在则下载
    if (!FileSystem::isFile(modelPath)) {
        if (!m_config->m_skipDownload) {
            bool downloadSuccess = Downloader::download(modelUrl, downloadDir);
            if (!downloadSuccess) {
                m_logger->error(std::format("[FaceEnhancer] Failed to download the model file: {}", modelUrl));
                return false;
            }
        } else {
            m_logger->error(std::format("[FaceEnhancer] Model file is not Found: {}", modelPath));
            return false;
        }
    }
    return true;
}

bool FaceEnhancer::preProcess(/*const std::unordered_set<std::string> &processMode*/) {
    m_logger->info("[FaceEnhancer] pre process");

    if (m_faceEnhancerModel == nullptr || *m_faceEnhancerModel != m_config->m_faceEnhancerModel) {
        if (m_faceEnhancerModel != nullptr && *m_faceEnhancerModel != m_config->m_faceEnhancerModel) {
            postCheck();
        }
        m_faceEnhancerModel = std::make_shared<Typing::EnumFaceEnhancerModel>(m_config->m_faceEnhancerModel);
        std::string modelPath = m_modelsInfoJson->at("faceEnhancerModels").at(m_modelName).at("path");

        this->createSession(modelPath);
        init();
    }

    return true;
}

Typing::VisionFrame FaceEnhancer::getReferenceFrame(const Face &sourceFace, const Face &targetFace, const VisionFrame &tempVisionFrame) {
    return *enhanceFace(targetFace, tempVisionFrame);
}

void FaceEnhancer::processImage(const std::unordered_set<std::string> &sourcePaths,
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
            m_logger->error("[FaceEnhancer] You must provide at least one reference face.");
            std::exit(1);
        }
    }

    Typing::VisionFrame targetFrame = Vision::readStaticImage(targetPath);

    m_logger->info("[FaceEnhancer] Processing image: " + targetPath);
    auto result = processFrame(referenceFaces, targetFrame);
    if (result == nullptr) {
        m_logger->error("[FaceEnhancer] Process image failed!");
        return;
    }
    Vision::writeImage(*result, outputPath);
    m_logger->info("[FaceEnhancer] Image processed successfully!");
}

void FaceEnhancer::processImages(const std::unordered_set<std::string> &sourcePaths,
                                 const std::vector<std::string> &targetPaths,
                                 const std::vector<std::string> &outputPaths) {
    if (targetPaths.size() != outputPaths.size()) {
        m_logger->error("[FaceEnhancer] The number of target paths and output paths must be equal");
        throw std::invalid_argument("[FaceEnhancer] The number of target paths and output paths must be equal");
    }
    if (targetPaths.empty()) {
        m_logger->error("[FaceEnhancer] No target paths provided!");
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
            m_logger->error("[FaceEnhancer] You must provide at least one reference face.");
            std::exit(1);
        }
    }

    dp::thread_pool poolForProcess(m_config->m_executionThreadCount);
    dp::thread_pool poolForWriteImage(std::thread::hardware_concurrency());
    std::vector<std::future<bool>> writeImageResults;
    for (int i = 0; i < targetPaths.size(); ++i) {
        auto targetFrame = Ffc::Vision::readStaticImage(targetPaths[i]);
        poolForProcess.enqueue([referenceFaces, targetFrame, &poolForWriteImage,
                                &writeImageResults, outputPath = outputPaths[i], this]() -> void {
            auto frame = processFrame(referenceFaces, targetFrame);
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
    bar.setPrefixText("[FaceEnhancer] Processing images");
    bar.setPostfixText(std::format("{}/{}", 0, targetPaths.size()));
    bar.setProgress(0);
    int i = 0;
    bool isAllWriteSuccess = true;
    while (true) {
        if (writeImageResults.size() <= i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        } else {
            if (!writeImageResults[i].valid()) {
                isAllWriteSuccess = false;
                m_logger->error(std::format("[FaceEnhancer] Failed to process image: {}", targetPaths[i]));
                ++i;
                continue;
            }

            auto writeIsSuccess = writeImageResults[i].get();
            if (!writeIsSuccess) {
                isAllWriteSuccess = false;
                m_logger->error(std::format("[FaceEnhancer] Failed to write image: {}", outputPaths[i]));
            }

            bar.setPostfixText(std::format("{}/{}", (i + 1), numTargetPaths));
            int progress = static_cast<int>(std::floor(((float)(i + 1) * 100.0f) / (float)numTargetPaths));
            bar.setProgress(progress);

            ++i;
            if (i >= outputPaths.size()) {
                break;
            }
        }
    }
    show_console_cursor(true);
    if (!isAllWriteSuccess) {
        m_logger->error("[FaceEnhancer] Some images failed to process or write.");
    }
}
} // namespace Ffc
