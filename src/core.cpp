/**
 ******************************************************************************
 * @file           : core.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-19
 ******************************************************************************
 */

#include "core.h"

namespace Ffc {
Core::Core() {
    m_env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    m_config = std::make_shared<Ffc::Config>("./facefusion.ini");
    m_logger = Logger::getInstance();
    m_modelsInfoJson = std::make_shared<nlohmann::json>();
    std::ifstream file("./modelsInfo.json");
    if (file.is_open()) {
        file >> *m_modelsInfoJson;
        file.close();
    }
    m_faceAnalyser = std::make_shared<Ffc::FaceAnalyser>(m_env, m_modelsInfoJson, m_config);
    m_faceMasker = std::make_shared<Ffc::FaceMasker>(m_env, m_modelsInfoJson, m_config);
}

void Core::forceDownload() {
    std::unordered_set<std::string> modelUrls;

    auto faceAnalyserModels = m_modelsInfoJson->at("faceAnalyserModels");
    for (const auto &model : faceAnalyserModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceMaskerModels = m_modelsInfoJson->at("faceMaskerModels");
    for (const auto &model : faceMaskerModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceSwapperModels = m_modelsInfoJson->at("faceSwapperModels");
    for (const auto &model : faceSwapperModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    auto faceEnhancerModels = m_modelsInfoJson->at("faceEnhancerModels");
    for (const auto &model : faceEnhancerModels) {
        modelUrls.insert(model.at("url").get<std::string>());
    }
    Downloader::batchDownload(modelUrls, "./models");
}

void Core::run() {
    m_logger->setLogLevel(m_config->m_logLevel);

    if (m_config->m_forceDownload) {
        this->forceDownload();
    }

    if (!preCheck() || !m_faceAnalyser->preCheck() || !m_faceMasker->preCheck()) {
        return;
    }

    // Start the frame processors
    for (const auto &processor : *getFrameProcessors()) {
        processor->preCheck();
    }

    conditionalProcess();
}

void Core::conditionalProcess() {
    auto startTime = std::chrono::steady_clock::now();
    for (const auto &processor : *getFrameProcessors()) {
        while (!processor->postCheck()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        processor->preProcess({"output"});
    }
    conditionalAppendReferenceFaces();
    if (FileSystem::hasImage(m_config->m_targetPaths)) {
        processImages(startTime);
    }
}

std::shared_ptr<std::vector<std::shared_ptr<ProcessorBase>>> Core::getFrameProcessors() {
    if (m_frameProcessors == nullptr) {
        m_frameProcessors = std::make_shared<std::vector<std::shared_ptr<ProcessorBase>>>();
        for (auto &processor : m_config->m_frameProcessors) {
            switch (processor) {
            case Typing::EnumFrameProcessor::FaceSwapper:
                m_frameProcessors->emplace_back(std::make_shared<Ffc::FaceSwapper>(m_env, m_faceAnalyser, m_faceMasker, m_modelsInfoJson, m_config));
                break;
            case Typing::EnumFrameProcessor::FaceEnhancer:
                m_frameProcessors->emplace_back(std::make_shared<Ffc::FaceEnhancer>(m_env, m_faceAnalyser, m_faceMasker, m_modelsInfoJson, m_config));
                break;
            }
        }
    }
    return m_frameProcessors;
}

bool Core::preCheck() const {
    // Todo do something
    return true;
}

void Core::conditionalAppendReferenceFaces() {
    if (m_config->m_faceSelectorMode == Typing::EnumFaceSelectorMode::FS_Reference
        && m_faceStore->getReferenceFaces().empty()) {
        std::unordered_set<std::string> sourcePaths = m_config->m_sourcePaths;
        if (sourcePaths.empty()) {
            m_logger->error("[Core] No image found in the source paths.");
            return;
        }

        if (m_config->m_referenceFacePath.empty()) {
            m_logger->error("[Core] Reference face path is empty.");
            return;
        }

        if (FileSystem::isImage(m_config->m_referenceFacePath)) {
            m_logger->error("[Core] Reference face path is not a valid image file.");
            return;
        }

        auto sourceFrame = Vision::readStaticImages(sourcePaths);
        auto sourceAverageFace = m_faceAnalyser->getAverageFace(sourceFrame);
        auto referenceFrame = Vision::readStaticImage(m_config->m_referenceFacePath);
        auto referenceFace = m_faceAnalyser->getOneFace(referenceFrame);
        m_faceStore->appendReferenceFace("origin", *referenceFace);
        if (!sourceAverageFace->isEmpty() && !referenceFace->isEmpty()) {
            for (const auto &processor : *getFrameProcessors()) {
                auto abstractFrame = processor->getReferenceFrame(*sourceAverageFace, *referenceFace, referenceFrame);
                if (!abstractFrame.empty()) {
                    m_faceStore->appendReferenceFace(typeid(*processor).name(), *sourceAverageFace);
                }
            }
        } else {
            m_logger->error("[Core] Source face or reference face is empty.");
        }
    }
}

void Core::processImages(const std::chrono::time_point<std::chrono::steady_clock> &startTime) {
    auto targetImagePathsSet = FileSystem::filterImagePaths(m_config->m_targetPaths);
    std::vector<std::string> targetImagePaths(targetImagePathsSet.begin(), targetImagePathsSet.end());
    targetImagePathsSet.clear();

    for (size_t i = 0; i < targetImagePaths.size(); ++i) {
        m_logger->info(std::format("[Core] Start processing image {}/{}: {}", i + 1, targetImagePaths.size(), targetImagePaths[i]));
        std::chrono::steady_clock::time_point tempStartTime = std::chrono::steady_clock::now();
        processImage(targetImagePaths[i], tempStartTime);
        std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - tempStartTime;
        m_logger->info(std::format("[Core] Finish processing image {}/{} Use {} seconds", i + 1, targetImagePaths.size(), elapsedSeconds));
    }
    FileSystem::removeDirectory(FileSystem::getTempPath());
    m_logger->info(std::format("[Core] All images processed successfully."));
}

void Core::processImage(const std::string &imagePath,
                        const std::chrono::time_point<std::chrono::steady_clock> &startTime) {
    std::string tempPath = FileSystem::getTempPath();
    const auto &targetImagePath = imagePath;
    if (FileSystem::copyImageToTemp(targetImagePath, m_config->m_outputImageResolution)) {
        m_logger->debug("[Core] Copy target image to temp path successfully.");
    } else {
        m_logger->error("[Core] Copy target image to temp path failed.");
        return;
    }
    auto tempTargetImagePath = tempPath + "/" + FileSystem::getFileName(targetImagePath);
    for (const auto &processor : *getFrameProcessors()) {
        processor->processImage(m_config->m_sourcePaths, tempTargetImagePath, tempTargetImagePath);
    }

    std::string normedOutputPath = FileSystem::normalizeOutputPath(targetImagePath, m_config->m_outputPath);
    if (FileSystem::finalizeImage(tempTargetImagePath, normedOutputPath, m_config->m_outputImageResolution, m_config->m_outputImageQuality)) {
        m_logger->debug("[Core] Finalize image successfully.");
    } else {
        m_logger->warn("[Core] Finalize image failed.");
    }
    FileSystem::removeFile(tempTargetImagePath);
    if (FileSystem::isImage(normedOutputPath)) {
        m_logger->info(std::format("[Core] Process image successfully. Output path: {}", normedOutputPath));
    } else {
        m_logger->error("[Core] Process image failed.");
    }
}
} // namespace Ffc