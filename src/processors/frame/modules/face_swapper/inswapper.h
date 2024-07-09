/**
 ******************************************************************************
 * @file           : inswapper.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-9
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_INSWAPPER_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_INSWAPPER_H_

#include <onnx/onnx_pb.h>
#include <fstream>
#include "face_swapper_base.h"
#include "face_swapper_session.h"

namespace Ffc {

class Inswapper : public FaceSwapperSession, public FaceSwapperBase {
public:
    Inswapper(const std::shared_ptr<Ort::Env> &env);
    ~Inswapper() override = default;
    std::shared_ptr<Typing::VisionFrame> applySwap(const Typing::Face &sourceFace,
                                                   const Typing::Face &targetFace,
                                                   const Typing::VisionFrame &targetFrame);

private:
    void preProcess();

    const std::string m_warpTemplate = "arcface_128_v2";
    const cv::Size m_size = cv::Size(128, 128);
    const std::vector<float> m_mean = {0.0, 0.0, 0.0, 0.0};
    const std::vector<float> m_standardDeviation = {1.0, 1.0, 1.0};
    int m_inputHeight;
    int m_inputWidth;
    std::vector<float> m_initializerArray;
    const int m_lenFeature = 512;
    std::vector<float> m_inputImageData;
    std::vector<float> m_inputEmbeddingData;
    
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_INSWAPPER_H_
