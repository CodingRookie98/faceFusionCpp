#include <iostream>
#include <nlohmann/json.hpp>
#include "processors/frame/modules/face_swapper.h"
#include "processors/frame/modules/face_enhancer.h"
#include "face_analyser/face_analyser.h"
#include "face_masker.h"
#include "config.h"

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    std::string sourcePath1 = "../../test/sources/YCY_1.jpg";
    std::string sourcePath2 = "../../test/sources/YCY_2.jpg";
    std::string sourcePath3 = "../../test/sources/YCY_3.jpg";
    std::string targetPath = "../../test/targets/target.jpg";
    std::string target2FacesPath = "../../test/targets/target_2faces.jpg";
    std::string swapOutputPath = "../../test/resultSwap_uniface_256_1.jpg";
    std::string enhanceOutputPath = "../../test/resultEnhance.jpg";

    const std::shared_ptr<Ffc::Config> config = std::make_shared<Ffc::Config>("./facefusion.ini");

//    config->m_sourcePaths.push_back(sourcePath1);
//    config->m_sourcePaths.push_back(sourcePath2);
//    config->m_sourcePaths.push_back(sourcePath3);

    auto modelsInfoJson = std::make_shared<nlohmann::json>();
    std::ifstream file("./modelsInfo.json");
    if (file.is_open()) {
        file >> *modelsInfoJson;
        file.close();
    }
    auto faceAnalyser = std::make_shared<Ffc::FaceAnalyser>(env, modelsInfoJson, config);
    auto faceMasker = std::make_shared<Ffc::FaceMasker>(env, modelsInfoJson, config);

    //    config->m_faceDetectorModelSet.clear();
    //    config->m_faceDetectorModelSet.insert(Ffc::Typing::EnumFaceDetectModel::FD_Yoloface);
    //    config->m_faceDetectorModelSet.insert(Ffc::Typing::EnumFaceDetectModel::FD_Scrfd);

    Ffc::FaceSwapper faceSwapper(env, faceAnalyser, faceMasker, modelsInfoJson, config);
    faceSwapper.processImage(config->m_sourcePaths, targetPath, swapOutputPath);

//    Ffc::FaceEnhancer faceEnhancer(env, faceAnalyser, faceMasker, modelsInfoJson, config);
//    faceEnhancer.processImage(swapOutputPath, enhanceOutputPath);

    return 0;
}
