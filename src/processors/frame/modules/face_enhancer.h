/**
 ******************************************************************************
 * @file           : face_enhancer.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-11
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_ENHANCER_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_ENHANCER_H_

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>
#include <thread_pool/thread_pool.h>
#include "config.h"
#include "typing.h"
#include "vision.h"
#include "face_analyser/face_analyser.h"
#include "face_masker.h"
#include "ort_session.h"
#include "processor_base.h"
#include "logger.h"
#include "face_store.h"
#include "progress_bar.h"

namespace Ffc {
class FaceEnhancer : private OrtSession, public ProcessorBase {
public:
    FaceEnhancer(const std::shared_ptr<Ort::Env> &env,
                 const std::shared_ptr<FaceAnalyser> &faceAnalyser,
                 const std::shared_ptr<FaceMasker> &faceMasker,
                 const std::shared_ptr<nlohmann::json> &modelsInfoJson,
                 const std::shared_ptr<const Config> &config);
    ~FaceEnhancer() override = default;
    bool preCheck() override;
    bool postCheck() override;
    bool preProcess(const std::unordered_set<std::string> &processMode) override;
    Typing::VisionFrame getReferenceFrame(const Typing::Face &sourceFace,
                                          const Typing::Face &targetFace,
                                          const Typing::VisionFrame &tempVisionFrame) override;

    void processImage(const std::unordered_set<std::string> &sourcePaths,
                      const std::string &targetPath,
                      const std::string &outputPath) override;
    void processImages(const std::unordered_set<std::string> &sourcePaths,
                       const std::vector<std::string> &targetPaths,
                       const std::vector<std::string> &outputPaths) override;

private:
    void init();
    std::shared_ptr<Typing::VisionFrame> processFrame(const Typing::Faces &referenceFaces,
                                                      const Typing::VisionFrame &targetFrame);
    std::shared_ptr<Typing::VisionFrame> enhanceFace(const Typing::Face &targetFace,
                                                     const Typing::VisionFrame &tempVisionFrame);
    std::shared_ptr<Typing::VisionFrame> blendFrame(const Typing::VisionFrame &targetFrame,
                                                    const Typing::VisionFrame &pasteVisionFrame);
    std::shared_ptr<Typing::VisionFrame> applyEnhance(const Typing::Face &targetFace,
                                                      const Typing::VisionFrame &tempVisionFrame);

    int m_inputHeight;
    int m_inputWidth;
    std::shared_ptr<FaceAnalyser> m_faceAnalyser;
    std::shared_ptr<FaceMasker> m_faceMasker;
    const std::shared_ptr<nlohmann::json> m_modelsInfoJson;
    const std::shared_ptr<const Config> m_config;
    std::string m_modelName;
    std::vector<cv::Point2f> m_warpTemplate;
    cv::Size m_size;
    std::shared_ptr<Typing::EnumFaceEnhancerModel> m_faceEnhancerModel;
    std::shared_ptr<Logger> m_logger = Logger::getInstance();
    std::shared_ptr<FaceStore> m_faceStore = FaceStore::getInstance();
};
} // namespace Ffc
#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_ENHANCER_H_
