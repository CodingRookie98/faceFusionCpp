/**
 ******************************************************************************
 * @file           : data_types.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-4
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_TYPING_H_
#define FACEFUSIONCPP_SRC_TYPING_H_

#include <vector>
#include <opencv2/imgproc.hpp>
#include <unordered_set>

namespace Ffc {
namespace Typing {

typedef struct BoudingBox {
    float xmin;
    float ymin;
    float xmax;
    float ymax;
} BoundingBox;

typedef std::vector<float> Embedding;
typedef std::vector<cv::Point2f> FaceLandmark;
typedef float Score;

typedef struct Face {
    BoundingBox boundingBox;
    FaceLandmark faceLandmark5;
    FaceLandmark faceLandmark68;
    FaceLandmark faceLandMark5_68;
    FaceLandmark faceLandmark68_5;
    Embedding embedding;
    Embedding normedEmbedding;
    Score detectorScore;
    Score landmarkerScore;
    int gender;
    int age;

    bool isEmpty() const {
        return faceLandmark68.empty();
    }
} Face;

typedef std::vector<Face> Faces;

typedef cv::Mat VisionFrame;

typedef std::tuple<int, int, int, int> Padding;
enum EnumFaceMaskRegion {
    All = 0,
    Skin = 1,
    LeftEyebrow = 2,
    RightEyebrow = 3,
    LeftEye = 4,
    RightEye = 5,
    Glasses = 6,
    Nose = 10,
    Mouth = 11,
    UpperLip = 12,
    LowerLip = 13
};
static const std::unordered_set<EnumFaceMaskRegion> faceMaskRegionAllSet = {
    EnumFaceMaskRegion::Skin,
    EnumFaceMaskRegion::LeftEyebrow,
    EnumFaceMaskRegion::RightEyebrow,
    EnumFaceMaskRegion::LeftEye,
    EnumFaceMaskRegion::RightEye,
    EnumFaceMaskRegion::Glasses,
    EnumFaceMaskRegion::Nose,
    EnumFaceMaskRegion::Mouth,
    EnumFaceMaskRegion::UpperLip,
    EnumFaceMaskRegion::LowerLip};

static const std::unordered_map<std::string, const EnumFaceMaskRegion> faceMaskRegionMap = {
    {"All", EnumFaceMaskRegion::All},
    {"skin", EnumFaceMaskRegion::Skin},
    {"left-eyebrow", EnumFaceMaskRegion::LeftEyebrow},
    {"right-eyebrow", EnumFaceMaskRegion::RightEyebrow},
    {"left-eye", EnumFaceMaskRegion::LeftEye},
    {"right-eye", EnumFaceMaskRegion::RightEye},
    {"glasses", EnumFaceMaskRegion::Glasses},
    {"nose", EnumFaceMaskRegion::Nose},
    {"mouth", EnumFaceMaskRegion::Mouth},
    {"upper-lip", EnumFaceMaskRegion::UpperLip},
    {"lower-lip", EnumFaceMaskRegion::LowerLip}};
enum EnumFaceDetectModel {
    FD_Many,
    FD_Retinaface,
    FD_Scrfd,
    FD_Yoloface
};
enum EnumFaceSelectorMode {
    FS_Many,
    FS_One,
    FS_Reference
};
enum EnumFaceMaskerType {
    FM_Box,
    FM_Occlusion,
    FM_Region
};
enum EnumFrameProcessor {
    FaceSwapper,
    FaceEnhancer,
};
enum EnumFaceSwapperModel {
    InSwapper_128,
    InSwapper_128_fp16,
};
enum EnumFaceEnhancerModel {
    FE_Gfpgan_14,
    FE_CodeFormer,
};
}
} // namespace Ffc::Typing
#endif // FACEFUSIONCPP_SRC_TYPING_H_
