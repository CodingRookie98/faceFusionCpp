/**
 ******************************************************************************
 * @file           : ort_session.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-12
 ******************************************************************************
 */

#include "ort_session.h"

namespace Ffc {
OrtSession::OrtSession(const std::shared_ptr<Ort::Env> &env) {
    m_env = env;
    m_sessionOptions = Ort::SessionOptions();
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    auto availableProviders = Ort::GetAvailableProviders();
    m_availableProviders.insert(availableProviders.begin(), availableProviders.end());

    if (m_config->m_executionProviders.contains(Typing::EnumExecutionProvider::EP_TensorRT)) {
        appendProviderTensorrt();
    }
    if (m_config->m_executionProviders.contains(Typing::EnumExecutionProvider::EP_CUDA)) {
        appendProviderCUDA();
    }
}

void OrtSession::createSession(const std::string &modelPath) {
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    // windows
    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    m_session = std::make_shared<Ort::Session>(*m_env, wideModelPath.c_str(), m_sessionOptions);
#else
    // linux
    m_session = std::make_shared<Ort::Session>(Ort::Session(*m_env, modelPath.c_str(), m_sessionOptions));
#endif

    size_t numInputNodes = m_session->GetInputCount();
    size_t numOutputNodes = m_session->GetOutputCount();

    m_inputNames.reserve(numInputNodes);
    m_outputNames.reserve(numOutputNodes);
    m_inputNamesPtrs.reserve(numInputNodes);
    m_outputNamesPtrs.reserve(numOutputNodes);

    Ort::AllocatorWithDefaultOptions allocator;
    for (size_t i = 0; i < numInputNodes; i++) {
        m_inputNamesPtrs.push_back(std::move(m_session->GetInputNameAllocated(i, allocator)));
        m_inputNames.push_back(m_inputNamesPtrs[i].get());
        Ort::TypeInfo inputTypeInfo = m_session->GetInputTypeInfo(i);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        auto inputDims = inputTensorInfo.GetShape();
        m_inputNodeDims.push_back(inputDims);
    }
    for (size_t i = 0; i < numOutputNodes; i++) {
        m_outputNamesPtrs.push_back(std::move(m_session->GetOutputNameAllocated(i, allocator)));
        m_outputNames.push_back(m_outputNamesPtrs[i].get());
        Ort::TypeInfo outputTypeInfo = m_session->GetOutputTypeInfo(i);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        auto outputDims = outputTensorInfo.GetShape();
        m_outputNodeDims.push_back(outputDims);
    }
}

void OrtSession::appendProviderCUDA() {
    if (!m_availableProviders.contains("CUDAExecutionProvider")) {
        m_logger->error("CUDA execution provider is not available in your environment.");
        return;
    }
    m_cudaProviderOptions = std::make_shared<OrtCUDAProviderOptions>();
    m_cudaProviderOptions->device_id = m_config->m_executionDeviceId;
    if (m_config->m_perSessionGpuMemLimit > 0) {
        m_cudaProviderOptions->gpu_mem_limit = m_config->m_perSessionGpuMemLimit * (1 << 30);
    }
    m_sessionOptions.AppendExecutionProvider_CUDA(*m_cudaProviderOptions);
}

void OrtSession::appendProviderTensorrt() {
    if (!m_availableProviders.contains("TensorrtExecutionProvider")) {
        m_logger->error("TensorRT execution provider is not available in your environment.");
        return;
    }
    m_tensorrtProviderOptions = std::make_shared<OrtTensorRTProviderOptions>();
    if (m_config->m_enableTensorrtCache) {
        m_tensorrtProviderOptions->trt_engine_cache_enable = true;
        m_tensorrtProviderOptions->trt_engine_cache_path = "./trt_engine_cache_path";
    }
    if (m_config->m_enableTensorrtEmbedEngine) {
        // TODO: Implement TENSORRT_EMBED_ENGINE
    }
    if (m_config->m_perSessionGpuMemLimit > 0) {
        m_tensorrtProviderOptions->trt_max_workspace_size = m_config->m_perSessionGpuMemLimit * (1 << 30);
    }
    m_tensorrtProviderOptions->device_id = m_config->m_executionDeviceId;
    m_sessionOptions.AppendExecutionProvider_TensorRT(*m_tensorrtProviderOptions);
}
} // namespace Ffc
