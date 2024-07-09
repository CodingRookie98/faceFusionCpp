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
// face analyser
static float faceDetectorScore = 0.5;
static float faceLandmarkerScore = 0.5;
enum EnumFaceDetectModel {
    FD_Many,
    FD_Retinaface,
    FD_Scrfd,
    FD_Yoloface
};
static EnumFaceDetectModel faceDetectorModel = FD_Yoloface;
static cv::Size faceDetectorSize(640, 640);

// face selector
enum EnumFaceSelectorMode {
    FS_Many,
    FS_One,
    FS_Reference
};
static EnumFaceSelectorMode faceSelectorMode = FS_Many;

// face masker
enum enumFaceMaskerType {
    FM_Box,
    FM_Occlusion,
    FM_Region
};
static std::unordered_set<enumFaceMaskerType> faceMaskerTypeSet = {FM_Box};
static float faceMaskBlur = 0.3;
static Typing::Padding faceMaskPadding = {0, 0, 0, 0};

// Frame Processors
enum EnumFrameProcessor {
    FaceSwapper,
    FaceEnhancer,
};
static std::unordered_set<EnumFrameProcessor> frameProcessorSet = {FaceSwapper};
enum EnumFaceSwapperModel {
    InSwapper_128,
    InSwapper_128_fp16,
};
static EnumFaceSwapperModel faceSwapperModel = InSwapper_128;
} // namespace Ffc::Globals




#endif // FACEFUSIONCPP_SRC_GLOBALS_H_
