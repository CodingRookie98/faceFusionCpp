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
    std::lock_guard<std::mutex> lock(m_mutex);
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    // windows
    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    try {
        m_session = std::make_shared<Ort::Session>(*m_env, wideModelPath.c_str(), m_sessionOptions);
    } catch (const Ort::Exception &e) {
        m_logger->error(std::format("CreateSession: Ort::Exception: {}", e.what()));
        return;
    } catch (const std::exception &e) {
        m_logger->error(std::format("CreateSession: std::exception: {}", e.what()));
        return;
    } catch (...) {
        m_logger->error("CreateSession: Unknown exception occurred.");
        return;
    }
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
    std::vector<const char *> keys;
    std::vector<const char *> values;
    const auto &api = Ort::GetApi();
    OrtTensorRTProviderOptionsV2 *tensorrtProviderOptionsV2;
    api.CreateTensorRTProviderOptions(&tensorrtProviderOptionsV2);

    std::string trtMaxWorkSpaceSiz;
    if (m_config->m_perSessionGpuMemLimit > 0) {
        trtMaxWorkSpaceSiz = std::to_string((size_t)m_config->m_perSessionGpuMemLimit * (1 << 30));
        keys.emplace_back("trt_max_workspace_size");
        values.emplace_back(trtMaxWorkSpaceSiz.c_str());
    }

    std::string deviceId = std::to_string(m_config->m_executionDeviceId);
    keys.emplace_back("device_id");
    values.emplace_back(deviceId.c_str());

    std::string enableTensorrtCache;
    std::string enableTensorrtEmbedEngine;
    std::string tensorrtEmbedEnginePath;
    if (m_config->m_enableTensorrtEmbedEngine) {
        enableTensorrtCache = std::to_string(m_config->m_enableTensorrtCache);
        enableTensorrtEmbedEngine = std::to_string(m_config->m_enableTensorrtEmbedEngine);
        tensorrtEmbedEnginePath = "./trt_engine_cache";

        keys.emplace_back("trt_engine_cache_enable");
        values.emplace_back(enableTensorrtCache.c_str());
        keys.emplace_back("trt_dump_ep_context_model");
        values.emplace_back(enableTensorrtEmbedEngine.c_str());
        keys.emplace_back("trt_ep_context_file_path");
        values.emplace_back(tensorrtEmbedEnginePath.c_str());
    }

    std::string tensorrtCachePath;
    if (m_config->m_enableTensorrtCache) {
        if (enableTensorrtEmbedEngine.empty()) {
            keys.emplace_back("trt_engine_cache_enable");
            values.emplace_back("1");
            tensorrtCachePath = "./trt_engine_cache/trt_engines";
        } else {
            tensorrtCachePath = "trt_engines";
        }

        keys.emplace_back("trt_engine_cache_path");
        values.emplace_back(tensorrtCachePath.c_str());
    }

    try {
        Ort::ThrowOnError(api.UpdateTensorRTProviderOptions(tensorrtProviderOptionsV2,
                                                            keys.data(), values.data(), keys.size()));
        m_sessionOptions.AppendExecutionProvider_TensorRT_V2(*tensorrtProviderOptionsV2);
    } catch (const Ort::Exception &e) {
        m_logger->error(std::format("Failed to append TensorRT execution provider: {}", e.what()));
    } catch (const std::exception &e) {
        m_logger->error(std::format("Failed to append TensorRT execution provider: {}", e.what()));
    } catch (...) {
        m_logger->error("Failed to append TensorRT execution provider: Unknown error");
    }
    api.ReleaseTensorRTProviderOptions(tensorrtProviderOptionsV2);
}
} // namespace Ffc
