/**
 ******************************************************************************
 * @file           : face_swapper.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#include "face_swapper.h"

namespace Ffc {
FaceSwapper::FaceSwapper(const std::shared_ptr<Ort::Env> &env) {
    m_env = env;
    // Todo
    m_faceAnalyser = std::make_shared<FaceAnalyser>(m_env);
}

void FaceSwapper::processImage(const std::vector<std::string> &sourcePaths,
                               const std::string &targetPath,
                               const std::string &outputPath) {
    Typing::Faces referenceFaces;
    std::vector<cv::Mat> sourceFrames = Ffc::Vision::readStaticImages(sourcePaths);
    std::shared_ptr<Typing::Face> sourceFace = m_faceAnalyser->getAverageFace(sourceFrames);
    auto targetFrame = Ffc::Vision::readStaticImage(targetPath);

    auto resultFrame = processFrame(referenceFaces, *sourceFace, targetFrame);
    Ffc::Vision::writeImage(*resultFrame, outputPath);
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::processFrame(const Typing::Faces &referenceFaces,
                                                               const Face &sourceFace,
                                                               const VisionFrame &targetFrame) {
    if (m_faceAnalyser == nullptr) {
        throw std::runtime_error("Face analyser is not set");
    }
    
    std::shared_ptr<Typing::VisionFrame> resultFrame = std::make_shared<Typing::VisionFrame>(targetFrame);
    if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_Many) {
        auto manyTargetFaces = m_faceAnalyser->getManyFaces(targetFrame);
        if (!manyTargetFaces->empty()) {
            for (auto &targetFace : *manyTargetFaces) {
                resultFrame = swapFace(sourceFace, targetFace, *resultFrame);
            }
        }
    } else if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_One) {
        // Todo
    } else if (Globals::faceSelectorMode == Globals::EnumFaceSelectorMode::FS_Reference) {
        // Todo
    }

    return resultFrame;
}

std::shared_ptr<Typing::VisionFrame> FaceSwapper::swapFace(const Face &sourceFace,
                                                           const Face &targetFace,
                                                           const VisionFrame &targetFrame) {
    if (m_swapperBase == nullptr || m_faceSwapperModel != Globals::faceSwapperModel) {
        switch (Globals::faceSwapperModel) {
        case Globals::InSwapper_128:
        case Globals::InSwapper_128_fp16:
            m_faceSwapperModel = Globals::faceSwapperModel;
            m_swapperBase = std::make_shared<Inswapper>(m_env);
            break;
        default:
            break;
        }
    }
    
    auto realSwapper = std::dynamic_pointer_cast<Inswapper>(m_swapperBase);
    auto resultFrame = realSwapper->applySwap(sourceFace, targetFace, targetFrame);
    return resultFrame;
}

void FaceSwapper::setFaceAnalyser(const std::shared_ptr<FaceAnalyser> &faceAnalyser) {
    m_faceAnalyser = faceAnalyser;
}

} // namespace Ffc