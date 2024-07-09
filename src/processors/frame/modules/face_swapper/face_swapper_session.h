/**
 ******************************************************************************
 * @file           : face_swapper_session.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-9
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_SESSION_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_SESSION_H_

#include "onnxruntime_cxx_api.h"
#include "iostream"

namespace Ffc {

class FaceSwapperSession {
public:
    FaceSwapperSession(const std::shared_ptr<Ort::Env> &env, const std::string &modelPath);
    ~FaceSwapperSession() = default;

protected:
    Ort::SessionOptions m_sessionOptions;
    std::shared_ptr<Ort::Env> m_env;
    std::shared_ptr<Ort::Session> m_session;
    std::vector<const char *> m_inputNames;
    std::vector<const char *> m_outputNames;
    std::vector<Ort::AllocatedStringPtr> m_inputNamesPtrs;
    std::vector<Ort::AllocatedStringPtr> m_outputNamesPtrs;
    std::vector<std::vector<int64_t>> m_inputNodeDims;  // >=1 outputs
    std::vector<std::vector<int64_t>> m_outputNodeDims; // >=1 outputs
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU);
    
private:
    void createSession(const std::string &modelPath);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_FACE_SWAPPER_FACE_SWAPPER_SESSION_H_
