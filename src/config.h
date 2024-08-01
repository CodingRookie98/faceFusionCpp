/**
 ******************************************************************************
 * @file           : config.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-17
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_CONFIG_H_
#define FACEFUSIONCPP_SRC_CONFIG_H_

#include <SimpleIni.h>
#include <string>
#include <vector>
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <shared_mutex>
#include "typing.h"
#include "file_system.h"
#include "vision.h"
#include "logger.h"

namespace Ffc {

class Config {
public:
    explicit Config(const std::string &configPath = "./facefusion.ini");
    ~Config() = default;
    Config(const Config &) = delete;
    Config &operator=(const Config &) = delete;
    Config(Config &&) = delete;
    Config &operator=(Config &&) = delete;

    static std::shared_ptr<Config> getInstance(const std::string &configPath = "./facefusion.ini");

    // general
    std::unordered_set<std::string> m_sourcePaths;
    std::unordered_set<std::string> m_targetPaths;
    std::string m_referenceFacePath;
    std::string m_outputPath;

    // misc
    bool m_forceDownload;
    bool m_skipDownload;
    Logger::LogLevel m_logLevel;
    
    // execution
    int m_executionDeviceId;
    std::unordered_set<Typing::EnumExecutionProvider> m_executionProviders;
    int m_executionThreadCount;

    // tensort
    bool m_enableTensorrtCache;
    bool m_enableTensorrtEmbedEngine;

    // memory
    float m_perSessionGpuMemLimit;

    // face analyser
    float m_faceDetectorScore;
    float m_faceLandmarkerScore;
    Typing::EnumFaceDetectModel m_faceDetectorModel;
    cv::Size m_faceDetectorSize;
    Typing::EnumFaceRecognizerModel m_faceRecognizerModel;

    // face selector
    Typing::EnumFaceSelectorMode m_faceSelectorMode;
    Typing::EnumFaceSelectorOrder m_faceSelectorOrder;
    Typing::EnumFaceSelectorAge m_faceSelectorAge;
    Typing::EnumFaceSelectorGender m_faceSelectorGender;
    int m_referenceFacePosition;
    float m_referenceFaceDistance;
    int m_referenceFrameNumber;

    // face masker
    std::unordered_set<Typing::EnumFaceMaskerType> m_faceMaskTypeSet;
    float m_faceMaskBlur;
    Typing::Padding m_faceMaskPadding;
    std::unordered_set<Typing::EnumFaceMaskRegion> m_faceMaskRegionsSet;

    // output creation
    int m_outputImageQuality;
    cv::Size m_outputImageResolution;

    // Frame Processors
    std::vector<Typing::EnumFrameProcessor>
        m_frameProcessors;
    Typing::EnumFaceSwapperModel m_faceSwapperModel;
    Typing::EnumFaceEnhancerModel m_faceEnhancerModel;
    int m_faceEnhancerBlend;

private:
    CSimpleIniA m_ini;
    std::shared_mutex m_sharedMutex;
    std::string m_configPath;
    std::shared_ptr<Logger> m_logger = Logger::getInstance();
    void loadConfig();
    static std::tuple<int, int, int, int> normalizePadding(const std::vector<int> &padding);
    static std::vector<int> parseStringToVector(const std::string &input);

    void general();
    void misc();
    void execution();
    void tensort();
    void memory();
    void faceAnalyser();
    void faceSelector();
    void faceMasker();
    void outputCreation();
    void frameProcessors();
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_CONFIG_H_
