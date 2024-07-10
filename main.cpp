#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include "processors/frame/modules/face_swapper/face_swapper.h"

int main() {
    std::shared_ptr<Ort::Env> env = std::make_shared<Ort::Env>(Ort::Env(ORT_LOGGING_LEVEL_ERROR, "faceFusionCpp"));
    std::string sourcePath1 = "../../test/YCY_1.jpg";
    std::string sourcePath2 = "../../test/YCY_2.jpg";
    std::string sourcePath3 = "../../test/YCY_3.jpg";
    std::string targetPath = "../../test/target.jpg";
    std::string outputPath = "../../test/result.jpg";

    std::vector<std::string> sourcePaths;
    sourcePaths.push_back(sourcePath1);
    sourcePaths.push_back(sourcePath2);
    sourcePaths.push_back(sourcePath3);

    Ffc::FaceSwapper faceSwapper(env);

    faceSwapper.processImage(sourcePaths, targetPath, outputPath);

    return 0;
}
