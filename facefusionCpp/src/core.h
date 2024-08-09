/**
 ******************************************************************************
 * @file           : core.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-19
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_CORE_H_
#define FACEFUSIONCPP_SRC_CORE_H_

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <chrono>
#include "ffmpeg_runner.h"
#include "processors/frame/modules/face_swapper.h"
#include "processors/frame/modules/face_enhancer.h"
#include "face_analyser/face_analyser.h"
#include "face_masker.h"
#include "config.h"
#include "logger.h"
#include "face_store.h"

namespace Ffc {
class Core {
public:
    Core();
    ~Core() = default;

    void run();
    void conditionalProcess();
    bool preCheck() const;
    void processImages(std::unordered_set<std::string> imagePaths);
    void processVideos(std::unordered_set<std::string> videoPaths);

private:
    std::shared_ptr<Ffc::Config> m_config;
    std::shared_ptr<Logger> m_logger;
    std::shared_ptr<Ort::Env> m_env;
    std::shared_ptr<nlohmann::json> m_modelsInfoJson;
    std::shared_ptr<FaceAnalyser> m_faceAnalyser;
    std::shared_ptr<FaceMasker> m_faceMasker;
    std::shared_ptr<std::vector<std::shared_ptr<ProcessorBase>>> m_frameProcessors;
    std::shared_ptr<FaceStore> m_faceStore = FaceStore::getInstance();

    void createFrameProcessors();
    std::shared_ptr<std::vector<std::shared_ptr<ProcessorBase>>> getFrameProcessors();
    void forceDownload();
    void conditionalAppendReferenceFaces();
    bool processVideo(const std::string &videoPath, const std::string &outputVideoPath);
    bool processVideoInSegments(const std::string &videoPath, const std::string &outputVideoPath, const unsigned int &duration);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_CORE_H_
