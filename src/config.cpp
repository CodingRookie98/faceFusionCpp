/**
 ******************************************************************************
 * @file           : config.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-17
 ******************************************************************************
 */

#include "config.h"

namespace Ffc {
Config::Config(const std::string &configPath) {
    if (!configPath.empty() && FileSystem::fileExists(configPath)) {
        m_configPath = configPath;
        loadConfig();
    } else {
        throw std::runtime_error("Config file not found");
    }
}

void Config::loadConfig() {
    std::unique_lock<std::shared_mutex> lock(m_sharedMutex);

    m_ini.SetUnicode();
    SI_Error rc = m_ini.LoadFile(m_configPath.c_str());
    if (rc < 0) {
        throw std::runtime_error("Failed to load config file");
    };

    std::string value;
    // general
    value = m_ini.GetValue("general", "source_dir_or_path", "");
    if (!value.empty()) {
        if (FileSystem::fileExists(value) && FileSystem::isFile(value) && FileSystem::isImage(value)) {
            m_sourcePaths.insert(value);
        } else if (FileSystem::isDirectory(value)) {
            m_sourcePaths = FileSystem::listFilesInDirectory(value);
        } else {
            throw std::runtime_error("source_dir_or_path is not a valid path or directory");
        }
    } else {
        throw std::runtime_error("source_dir_or_path is not set");
    }
    value = m_ini.GetValue("general", "target_dir_or_path", "");
    if (!value.empty()) {
        if (FileSystem::fileExists(value) && FileSystem::isFile(value) && FileSystem::isImage(value)) {
            m_targetPaths.insert(value);
        } else if (FileSystem::isDirectory(value)) {
            m_targetPaths = FileSystem::listFilesInDirectory(value);
        } else {
            throw std::runtime_error("target_dir_or_path is not a valid path or directory");
        }
    } else {
        throw std::runtime_error("target_dir_or_path is not set");
    }
    value = m_ini.GetValue("general", "output_dir_or_path", "./output");
    if (!value.empty()) {
        m_outputPath = value;
    } else {
        m_outputPath = "./output";
    }

    // face_analyser
    value = m_ini.GetValue("face_analyser", "ace_analyser_model", "yoloface");
    if (!value.empty()) {
        if (value == "many") {
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Many);
        } else if (value == "retinaface") {
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Retina);
        } else if (value == "yoloface") {
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Yoloface);
        } else if (value == "scrfd") {
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Scrfd);
        } else if (value == "yunet") {
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Yunet);
        } else {
            std::cerr << "Invalid ace_analyser_model value: " << value << " Use default: yoloface" << std::endl;
            m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Yoloface);
        }
    } else {
        m_faceDetectorModelSet.insert(Typing::EnumFaceDetectModel::FD_Yoloface);
    }
    value = m_ini.GetValue("face_analyser", "face_detector_size", "640x640");
    if (!value.empty()) {
        m_faceDetectorSize = Vision::unpackResolution(value);
    } else {
        m_faceDetectorSize = cv::Size(640, 640);
    }
    value = m_ini.GetValue("face_analyser", "face_detector_score", "0.5");
    if (!value.empty()) {
        m_faceDetectorScore = std::stof(value);
        if (m_faceDetectorScore < 0.0f) {
            m_faceDetectorScore = 0.0f;
        } else if (m_faceDetectorScore > 1.0f) {
            m_faceDetectorScore = 1.0f;
        }
    }else {
        m_faceDetectorScore = 0.5f;
    }
    value = m_ini.GetValue("face_analyser", "face_landmaker_score", "0.5");
    if (!value.empty()) {
        m_faceLandmarkerScore = std::stof(value);
    } else {
        m_faceLandmarkerScore = 0.5f;
    }

    // face_selector
    value = m_ini.GetValue("face_selector", "face_selector_mode", "reference");
    if (!value.empty()) {
        if (value == "reference") {
            m_faceSelectorMode = Typing::EnumFaceSelectorMode::FS_Reference;
        } else if (value == "one") {
            m_faceSelectorMode = Typing::EnumFaceSelectorMode::FS_One;
        } else if (value == "many") {
            m_faceSelectorMode = Typing::EnumFaceSelectorMode::FS_Many;
        } else {
            std::cerr << "Invalid face_selector_mode value: " << value << " Use default: reference" << std::endl;
            m_faceSelectorMode = Typing::EnumFaceSelectorMode::FS_Reference;
        }
    } else {
        m_faceSelectorMode = Typing::EnumFaceSelectorMode::FS_Reference;
    }
    value = m_ini.GetValue("face_selector", "face_selector_order", "left-right");
    if (!value.empty()) {
        if (value == "left-right") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Left_Right;
        } else if (value == "right-left") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Right_Left;
        } else if (value == "top-bottom") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Top_Bottom;
        } else if (value == "bottom-top") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Bottom_Top;
        } else if (value == "smAll-large") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Small_Large;
        } else if (value == "large-smAll") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Large_Small;
        } else if (value == "best-worst") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Best_Worst;
        } else if (value == "worst-best") {
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Worst_Best;
        } else {
            std::cerr << "Invalid face_selector_order value: " << value << " Use default: left-right" << std::endl;
            m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Left_Right;
        }
    }else {
        m_faceSelectorOrder = Typing::EnumFaceSelectorOrder::FSO_Left_Right;
    }
    value = m_ini.GetValue("face_selector", "face_selector_age", "All");
    if (!value.empty()) {
        if (value == "child") {
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_Child;
        } else if (value == "teen") {
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_Teenager;
        } else if (value == "adult") {
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_Adult;
        } else if (value == "senior") {
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_Senior;
        } else if (value == "All") {
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_All;
        } else {
            std::cerr << "Invalid face_selector_age value: " << value << " Use default: All" << std::endl;
            m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_All;
        }
    } else {
        m_faceSelectorAge = Typing::EnumFaceSelectorAge::FSA_All;
    }
    value = m_ini.GetValue("face_selector", "face_selector_gender", "All");
    if (!value.empty()) {
        if (value == "male") {
            m_faceSelectorGender = Typing::EnumFaceSelectorGender::FSG_Male;
        } else if (value == "female") {
            m_faceSelectorGender = Typing::EnumFaceSelectorGender::FSG_Female;
        } else if (value == "All") {
            m_faceSelectorGender = Typing::EnumFaceSelectorGender::FSG_All;
        } else {
            std::cerr << "Invalid face_selector_gender value: " << value << " Use default: All" << std::endl;
            m_faceSelectorGender = Typing::EnumFaceSelectorGender::FSG_All;
        }
    } else {
        m_faceSelectorGender = Typing::EnumFaceSelectorGender::FSG_All;
    }
    value = m_ini.GetValue("face_selector", "reference_face_position", "0");
    if (!value.empty()) {
        m_referenceFacePosition = std::stoi(value);
        if (m_referenceFacePosition < 0) {
            m_referenceFacePosition = 0;
        }
    }else {
        m_referenceFacePosition = 0;
    }
    value = m_ini.GetValue("face_selector", "reference_face_distance", "0.6");
    if (!value.empty()) {
        m_referenceFaceDistance = std::stof(value);
        if (m_referenceFaceDistance < 0) {
            m_referenceFaceDistance = 0;
        } else if (m_referenceFaceDistance > 1.5) {
            m_referenceFaceDistance = 1.5;
        }
    }else {
        m_referenceFaceDistance = 0.6f;
    }
    value = m_ini.GetValue("face_selector", "reference_frame_number", "0");
    if (!value.empty()) {
        m_referenceFrameNumber = std::stoi(value);
        if (m_referenceFrameNumber < 0) {
            m_referenceFrameNumber = 0;
        }
    }else {
        m_referenceFrameNumber = 0;
    }

    // face_mask
    value = m_ini.GetValue("face_mask", "face_mask_types", "box");
    if (!value.empty()) {
        if (value.find("box") != std::string::npos) {
            m_faceMaskTypeSet.insert(Typing::EnumFaceMaskerType::FM_Box);
        }
        if (value.find("occlusion") != std::string::npos) {
            m_faceMaskTypeSet.insert(Typing::EnumFaceMaskerType::FM_Occlusion);
        }
        if (value.find("region") != std::string::npos) {
            m_faceMaskTypeSet.insert(Typing::EnumFaceMaskerType::FM_Region);
        }
    }else {
        m_faceMaskTypeSet.insert(Typing::EnumFaceMaskerType::FM_Box);
    }
    value = m_ini.GetValue("face_mask", "face_mask_blur", "0.3");
    if (!value.empty()) {
        m_faceMaskBlur = std::stof(value);
        if (m_faceMaskBlur < 0) {
            m_faceMaskBlur = 0;
        } else if (m_faceMaskBlur > 1) {
            m_faceMaskBlur = 1;
        }
    } else {
        m_faceMaskBlur = 0.3f;
    }
    value = m_ini.GetValue("face_mask", "face_mask_padding", "0 0 0 0");
    if (!value.empty()) {
        m_faceMaskPadding = normalizePadding(parseStringToVector(value));
    }else {
        m_faceMaskPadding = std::tuple(0, 0, 0, 0);
    }
    value = m_ini.GetValue("face_mask", "face_mask_region", "All");
    if (!value.empty()) {
        if (value == "All") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::All);
        } else if (value == "skin") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::Skin);
        } else if (value == "nose") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::Nose);
        } else if (value == "left-eyebrow") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::LeftEyebrow);
        } else if (value == "right-eyebrow") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::RightEyebrow);
        } else if (value == "mouth") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::Mouth);
        } else if (value == "right-eye") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::RightEye);
        } else if (value == "left-eye") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::LeftEye);
        } else if (value == "glasses") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::Glasses);
        } else if (value == "upper-lip") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::UpperLip);
        } else if (value == "lower-lip") {
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::LowerLip);
        } else {
            std::cerr <<  "Invalid face mask region: " << value << " Use default: All" << std::endl;
            m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::All);
        }
    } else {
        m_faceMaskRegionsSet.insert(Typing::EnumFaceMaskRegion::All);
    }

    // output_creation
    value = m_ini.GetValue("output_creation", "output_image_quality", "80");
    if (!value.empty()) {
        m_outputImageQuality = std::stoi(value);
        if (m_outputImageQuality < 0) {
            m_outputImageQuality = 0;
        } else if (m_outputImageQuality > 100) {
            m_outputImageQuality = 100;
        }
    }else {
        m_outputImageQuality = 80;
    }
    value = m_ini.GetValue("output_creation", "output_image_resolution", "");
    if (!value.empty()) {
        m_outputImageResolution = Vision::unpackResolution(value);
    } else {
        m_outputImageResolution = cv::Size(0, 0);
    }

    // frame_processors
    value = m_ini.GetValue("frame_processors", "frame_processors", "face_swapper");
    if (!value.empty()) {
        if (value.find("face_swapper") != std::string::npos) {
            m_frameProcessorSet.insert(Typing::EnumFrameProcessor::FaceSwapper);
        }
        if (value.find("face_enhancer") != std::string::npos) {
            m_frameProcessorSet.insert(Typing::EnumFrameProcessor::FaceEnhancer);
        }
    }else {
        m_frameProcessorSet.insert(Typing::EnumFrameProcessor::FaceSwapper);
    }
    value = m_ini.GetValue("frame_processors", "face_enhancer_model", "gfpgan_1.4");
    if (!value.empty()) {
        if (value == "codeformer") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_CodeFormer;
        } else if (value == "gfpgan_1.2") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gfpgan_12;
        } else if (value == "gfpgan_1.3") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gfpgan_13;
        } else if (value == "gfpgan_1.4") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gfpgan_14;
        } else if (value == "restoreformer_plus_plus") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Restoreformer_plus_plus;
        } else if (value == "gpen_bfr_256") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gpen_bfr_256;
        } else if (value == "gpen_bfr_512") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gpen_bfr_512;
        } else if (value == "gpen_bfr_1024") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gpen_bfr_1024;
        } else if (value == "gpen_bfr_2048") {
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gpen_bfr_2048;
        } else {
            std::cerr << "Invalid face enhancer model: " << value << " Use Default: gfpgan_1.4" << std::endl;
            m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gfpgan_14;
        }
    }else {
        m_faceEnhancerModel = Typing::EnumFaceEnhancerModel::FEM_Gfpgan_14;
    }
    value = m_ini.GetValue("frame_processors", "face_enhancer_blend", "80");
    if (!value.empty()) {
        m_faceEnhancerBlend = std::stoi(value);
        if (m_faceEnhancerBlend < 0) {
            m_faceEnhancerBlend = 0;
        } else if (m_faceEnhancerBlend > 100) {
            m_faceEnhancerBlend = 100;
        }
    }else {
        m_faceEnhancerBlend = 80;
    }
    value = m_ini.GetValue("frame_processors", "face_swapper_model", "inswapper_128_fp16");
    if (!value.empty()) {
        if (value == "inswapper_128_fp16") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Inswapper_128_fp16;
        } else if (value == "inswapper_128") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Inswapper_128;
        } else if (value == "simswap_256") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Simswap_256;
        } else if (value == "simswap_512_unofficial") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Simswap_512_unofficial;
        } else if (value == "uniface_256") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Uniface_256;
        } else if (value == "blendswap_256") {
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Blendswap_256;
        } else {
            std::cerr << "Invalid face swapper model: " << value << " Use Default: inswapper_128_fp16" << std::endl;
            m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Inswapper_128_fp16;
        }
    } else {
        m_faceSwapperModel = Typing::EnumFaceSwapperModel::FSM_Inswapper_128_fp16;
    }
}

std::tuple<int, int, int, int> Config::normalizePadding(const std::vector<int> &padding) {
    if (padding.size() == 1) {
        return std::make_tuple(padding[0], padding[0], padding[0], padding[0]);
    } else if (padding.size() == 2) {
        return std::make_tuple(padding[0], padding[1], padding[0], padding[1]);
    } else if (padding.size() == 3) {
        return std::make_tuple(padding[0], padding[1], padding[2], padding[1]);
    } else if (padding.size() == 4) {
        return std::make_tuple(padding[0], padding[1], padding[2], padding[3]);
    } else {
        throw std::invalid_argument("Invalid padding length");
    }
}

std::vector<int> Config::parseStringToVector(const std::string &input) {
    std::vector<int> values;
    std::stringstream ss(input);
    int value;
    while (ss >> value) {
        values.push_back(value);
    }
    if (ss.fail() && !ss.eof()) {
        throw std::invalid_argument("Invalid input format");
    }
    return values;
}
} // namespace Ffc