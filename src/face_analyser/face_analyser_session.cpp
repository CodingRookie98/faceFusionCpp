/**
 ******************************************************************************
 * @file           : face_analyser_session.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */


#include <iostream>
#include "face_analyser_session.h"
namespace Ffc {

FaceAnalyserSession::FaceAnalyserSession(const std::shared_ptr<Ort::Env> &env, const std::string &modelPath) {
    m_env = env;
    m_sessionOptions = Ort::SessionOptions();
    m_sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    createSession(modelPath);
}

void FaceAnalyserSession::createSession(const std::string &modelPath) {
    // CUDA 加速
    try {
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(m_sessionOptions, 0));
    } catch (const Ort::Exception &e) {
        std::cerr << "ONNX Runtime exception caught: " << e.what() << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Standard exception caught: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown exception caught" << std::endl;
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
    // windows
    std::wstring wideModelPath(modelPath.begin(), modelPath.end());
    m_session = std::make_shared<Ort::Session>(Ort::Session(*m_env, wideModelPath.c_str(), m_sessionOptions));
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
//    m_inputHeight = m_inputNodeDims[0][2];
//    m_inputWidth = m_inputNodeDims[0][3];
}
FaceAnalyserSession::~FaceAnalyserSession() = default;
}