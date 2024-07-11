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
#include "typing.h"
#include "vision.h"
#include "globals.h"
#include "face_analyser/face_analyser.h"
#include "face_masker.h"

namespace Ffc {
class FaceEnhancer {
public:
    FaceEnhancer(const std::shared_ptr<Ort::Env> &env);
    ~FaceEnhancer() = default;

    void processImage(const std::string &targetPath, const std::string &outputPath);
    void setFaceAnalyser(const std::shared_ptr<FaceAnalyser> &faceAnalyser);

private:
    void init();
    std::shared_ptr<Typing::VisionFrame> processFrame(const Typing::Faces &referenceFaces,
                                                      const Typing::VisionFrame &targetFrame);
    std::shared_ptr<Typing::VisionFrame > enhanceFace(const Typing::Face &targetFace,
                                                       const Typing::VisionFrame &tempVisionFrame);
 
    std::shared_ptr<Ort::Env> m_env;
    std::shared_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_sessionOptions;
    std::shared_ptr<OrtCUDAProviderOptions> m_cudaProviderOptions = nullptr;
    std::vector<const char *> m_inputNames;
    std::vector<const char *> m_outputNames;
    std::vector<Ort::AllocatedStringPtr> m_inputNamesPtrs;
    std::vector<Ort::AllocatedStringPtr> m_outputNamesPtrs;
    std::vector<std::vector<int64_t>> m_inputNodeDims;  // >=1 outputs
    std::vector<std::vector<int64_t>> m_outputNodeDims; // >=1 outputs
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    int m_inputHeight;
    int m_inputWidth;
    std::vector<float> m_inputImageData;
    std::shared_ptr<nlohmann::json> m_modelsJson;
    std::shared_ptr<FaceAnalyser> m_faceAnalyser;
    std::string m_modelName;
    std::vector<cv::Point2f > m_warpTemplate;
    cv::Size m_size;
    std::shared_ptr<Globals::EnumFaceEnhancerModel> m_faceEnhancerModel;
};
} // namespace Ffc
#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_ENHANCER_H_
