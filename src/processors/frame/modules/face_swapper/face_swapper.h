/**
 ******************************************************************************
 * @file           : face_swapper.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_H_

#include <onnxruntime_cxx_api.h>
#include "vision.h"
#include "typing.h"
#include "face_analyser/face_analyser.h"
#include "globals.h"
#include "face_masker.h"
#include "face_swapper_base.h"
#include "inswapper.h"

namespace Ffc {

class FaceSwapper {
public:
    explicit FaceSwapper(const std::shared_ptr<Ort::Env> &env);
    ~FaceSwapper() = default;

    void processImage(const std::vector<std::string> &sourcePaths,
                      const std::string &targetPath,
                      const std::string &outputPath);
    
 private:
    std::shared_ptr<Typing::VisionFrame> processFrame(const Typing::Faces &referenceFaces,
                                                      const Typing::Face &sourceFace,
                                                      const Typing::VisionFrame &targetFrame);
    std::shared_ptr<Typing::VisionFrame> swapFace(const Typing::Face &sourceFace, const Typing::Face &targetFace,
                                                  const Typing::VisionFrame &targetFrame);
    
    std::unique_ptr<FaceAnalyser> m_faceAnalyser;
    std::shared_ptr<Ort::Env> m_env;
    std::shared_ptr<FaceSwapperBase> m_swapperBase;
    Globals::EnumFaceSwapperModel m_faceSwapperModel = Globals::faceSwapperModel;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_H_
