/**
 ******************************************************************************
 * @file           : gloabals.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_GLOBALS_H_
#define FACEFUSIONCPP_SRC_GLOBALS_H_

#include <unordered_set>

namespace Ffc::Globals {
// general
static std::vector<std::string> sourcePaths;
static std::vector<std::string> targetPaths;
static std::string outputPath;

// face analyser
static float faceDetectorScore = 0.5;
static float faceLandmarkerScore = 0.5;
static Typing::EnumFaceDetectModel faceDetectorModel = Typing::EnumFaceDetectModel::FD_Yoloface;
static cv::Size faceDetectorSize(640, 640);

// face selector
static Typing::EnumFaceSelectorMode faceSelectorMode = Typing::EnumFaceSelectorMode::FS_Many;

// face masker
static std::unordered_set<Typing::EnumFaceMaskerType>
    faceMaskerTypeSet = {Typing::EnumFaceMaskerType::FM_Region};
static float faceMaskBlur = 0.3;
static Typing::Padding faceMaskPadding = {0, 0, 0, 0};
static std::unordered_set<Typing::EnumFaceMaskRegion>
    faceMaskRegionsSet = {Typing::EnumFaceMaskRegion::LeftEye,
                          Typing::EnumFaceMaskRegion::RightEye,
                          Typing::EnumFaceMaskRegion::Nose
                          /* Typing::EnumFaceMaskRegion::All*/};

// Frame Processors
static std::unordered_set<Typing::EnumFrameProcessor>
    frameProcessorSet = {Typing::EnumFrameProcessor::FaceSwapper};
static Typing::EnumFaceSwapperModel faceSwapperModel = Typing::EnumFaceSwapperModel::InSwapper_128;
static Typing::EnumFaceEnhancerModel faceEnhancerModel = Typing::EnumFaceEnhancerModel::FE_Gfpgan_14;
static int faceEnhancerBlend = 100;
} // namespace Ffc::Globals

#endif // FACEFUSIONCPP_SRC_GLOBALS_H_
