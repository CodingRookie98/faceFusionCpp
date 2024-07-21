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
        // Todo processor.preProcess
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
} // namespace Ffc