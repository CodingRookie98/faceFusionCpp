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
#include "ort_session.h"

namespace Ffc {
class FaceEnhancer : public OrtSession {
public:
    FaceEnhancer(const std::shared_ptr<Ort::Env> &env,
                 const std::shared_ptr<FaceAnalyser> &faceAnalyser,
                 const std::shared_ptr<FaceMasker> &faceMasker,
                 const std::shared_ptr<nlohmann::json> &modelsInfoJson);
    ~FaceEnhancer() = default;

    void processImage(const std::string &targetPath, const std::string &outputPath);
    void setFaceAnalyser(const std::shared_ptr<FaceAnalyser> &faceAnalyser);

private:
    void init();
    std::shared_ptr<Typing::VisionFrame> processFrame(const Typing::Faces &referenceFaces,
                                                      const Typing::VisionFrame &targetFrame);
    std::shared_ptr<Typing::VisionFrame> enhanceFace(const Typing::Face &targetFace,
                                                     const Typing::VisionFrame &tempVisionFrame);
    static std::shared_ptr<Typing::VisionFrame> blendFrame(const Typing::VisionFrame &targetFrame,
                                                    const Typing::VisionFrame &pasteVisionFrame);
    std::shared_ptr<Typing::VisionFrame> applyEnhance(const Typing::Face &targetFace,
                                                      const Typing::VisionFrame &tempVisionFrame);

    int m_inputHeight;
    int m_inputWidth;
    std::vector<float> m_inputImageData;
    std::shared_ptr<FaceAnalyser> m_faceAnalyser;
    std::shared_ptr<FaceMasker> m_faceMasker;
    std::shared_ptr<nlohmann::json> m_modelsJson;
    std::string m_modelName;
    std::vector<cv::Point2f> m_warpTemplate;
    cv::Size m_size;
    std::shared_ptr<Typing::EnumFaceEnhancerModel> m_faceEnhancerModel;
};
} // namespace Ffc
#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_ENHANCER_H_
