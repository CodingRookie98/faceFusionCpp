#include <iostream>
#include <vector>
#include <nlohmann/json.hpp>
#include "processors/frame/modules/face_swapper.h"
#include "processors/frame/modules/face_enhancer.h"
#include "face_analyser/face_analyser.h"
#include "face_masker.h"
#include "globals.h"
#include "downloader.h"

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    std::string sourcePath1 = "../../test/YCY_1.jpg";
    std::string sourcePath2 = "../../test/YCY_2.jpg";
    std::string sourcePath3 = "../../test/YCY_3.jpg";
    std::string targetPath = "../../test/target.jpg";
    std::string swapOutputPath = "../../test/resultSwap.jpg";
    std::string enhanceOutputPath = "../../test/resultEnhance.jpg";

    Ffc::Globals::sourcePaths.push_back(sourcePath1);
    Ffc::Globals::sourcePaths.push_back(sourcePath2);
    Ffc::Globals::sourcePaths.push_back(sourcePath3);

    auto modelsInfoJson = std::make_shared<nlohmann::json>();
    std::ifstream file("./modelsInfo.json");
    if (file.is_open()) {
        file >> *modelsInfoJson;
        file.close();
    }
    auto faceAnalyser = std::make_shared<Ffc::FaceAnalyser>(env, modelsInfoJson);
    auto faceMasker = std::make_shared<Ffc::FaceMasker>(env, modelsInfoJson);

    Ffc::FaceSwapper faceSwapper(env, faceAnalyser, faceMasker, modelsInfoJson);
    faceSwapper.processImage(Ffc::Globals::sourcePaths, targetPath, swapOutputPath);

    Ffc::FaceEnhancer faceEnhancer(env, faceAnalyser, faceMasker, modelsInfoJson);
    faceEnhancer.processImage(swapOutputPath, enhanceOutputPath);

    //    Ffc::Downloader::downloadFileFromURL("https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx",
    //                                      "./temp");

    return 0;
}
