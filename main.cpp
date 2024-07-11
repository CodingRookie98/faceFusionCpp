#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <nlohmann/json.hpp>
#include "processors/frame/modules/face_swapper/face_swapper.h"
#include "processors/frame/modules/face_enhancer.h"
#include "face_analyser/face_analyser.h"

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    std::string sourcePath1 = "../../test/YCY_1.jpg";
    std::string sourcePath2 = "../../test/YCY_2.jpg";
    std::string sourcePath3 = "../../test/YCY_3.jpg";
    std::string targetPath = "../../test/target.jpg";
    std::string swapOutputPath = "../../test/resultSwap.jpg";
    std::string enhanceOutputPath = "../../test/resultEnhance.jpg";
    

    std::vector<std::string> sourcePaths;
    sourcePaths.push_back(sourcePath1);
    sourcePaths.push_back(sourcePath2);
    sourcePaths.push_back(sourcePath3);

    auto faceAnalyser = std::make_shared<Ffc::FaceAnalyser>(env);

//    Ffc::FaceSwapper faceSwapper(env);
//    faceSwapper.processImage(sourcePaths, targetPath, swapOutputPath);
    
    Ffc::FaceEnhancer faceEnhancer(env);
    faceEnhancer.setFaceAnalyser(faceAnalyser);
    faceEnhancer.processImage(swapOutputPath, enhanceOutputPath);

    return 0;
}
