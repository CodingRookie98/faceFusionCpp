/**
 ******************************************************************************
 * @file           : data_types.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24_7_4
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
    {"left_eyebrow", EnumFaceMaskRegion::LeftEyebrow},
    {"right_eyebrow", EnumFaceMaskRegion::RightEyebrow},
    {"left_eye", EnumFaceMaskRegion::LeftEye},
    {"right_eye", EnumFaceMaskRegion::RightEye},
    {"glasses", EnumFaceMaskRegion::Glasses},
    {"nose", EnumFaceMaskRegion::Nose},
    {"mouth", EnumFaceMaskRegion::Mouth},
    {"upper_lip", EnumFaceMaskRegion::UpperLip},
    {"lower_lip", EnumFaceMaskRegion::LowerLip}};
enum EnumFaceDetectModel {
    FD_Many,
    FD_Retina,
    FD_Scrfd,
    FD_Yoloface,
    FD_Yunet
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
    FSM_Inswapper_128,
    FSM_Inswapper_128_fp16,
    FSM_Blendswap_256,
    FSM_Simswap_256,
    FSM_Simswap_512_unofficial,
    FSM_Uniface_256
};
enum EnumFaceEnhancerModel {
    FEM_CodeFormer,
    FEM_Gfpgan_12,
    FEM_Gfpgan_13,
    FEM_Gfpgan_14,
    FEM_Gpen_bfr_256,
    FEM_Gpen_bfr_512,
    FEM_Gpen_bfr_1024,
    FEM_Gpen_bfr_2048,
    FEM_Restoreformer_plus_plus
};

enum EnumFaceSelectorOrder {
    FSO_Left_Right,
    FSO_Right_Left,
    FSO_Top_Bottom,
    FSO_Bottom_Top,
    FSO_Small_Large,
    FSO_Large_Small,
    FSO_Best_Worst,
    FSO_Worst_Best
};
enum EnumFaceSelectorAge {
    FSA_All,
    FSA_Child,
    FSA_Teenager,
    FSA_Adult,
    FSA_Senior
};
enum EnumFaceSelectorGender {
    FSG_All,
    FSG_Male,
    FSG_Female
};
enum EnumFaceRecognizerModel {
    FRM_ArcFaceBlendSwap,
    FRM_ArcFaceInswapper,
    FRM_ArcFaceSimSwap,
    FRM_ArcFaceUniface
};
enum EnumExecutionProvider {
    EP_CPU,
    EP_CUDA,
    EP_TensorRT
};
}
} // namespace Ffc::Typing
#endif // FACEFUSIONCPP_SRC_TYPING_H_
