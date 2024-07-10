/**
 ******************************************************************************
 * @file           : face_swapper_base.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-9
 ******************************************************************************
 */

#include "face_swapper_base.h"

namespace Ffc {
std::shared_ptr<Typing::VisionFrame>
FaceSwapperBase::prepareCropVisionFrame(const Typing::VisionFrame &visionFrame,
                                        const std::vector<float> &mean,
                                        const std::vector<float> &standDeviation) {
    cv::Mat bgrImage = visionFrame.clone();
    std::vector<cv::Mat> bgrChannels(3);
    split(bgrImage, bgrChannels);
    for (int c = 0; c < 3; c++) {
        bgrChannels[c].convertTo(bgrChannels.at(c), CV_32FC1, 1 / (255.0 * standDeviation.at(c)),
                                 -mean.at(c) / (float)standDeviation.at(c));
    }

    cv::Mat processedBGR;
    cv::merge(bgrChannels, processedBGR);

    return std::make_shared<Typing::VisionFrame>(processedBGR);
}

std::shared_ptr<std::tuple<Typing::VisionFrame, cv::Mat>>
FaceSwapperBase::getCropVisionFrameAndAffineMat(const Typing::VisionFrame &visionFrame,
                                                const Typing::FaceLandmark &faceLandmark,
                                                const std::string &modelTemplate,
                                                const cv::Size &size) {
    return FaceHelper::warpFaceByFaceLandmarks5(visionFrame, faceLandmark, modelTemplate, size);
}

std::shared_ptr<std::list<cv::Mat>>
FaceSwapperBase::getCropMaskList(const Typing::VisionFrame &visionFrame,
                                 const cv::Size &cropSize, const float &faceMaskBlur,
                                 const Padding &faceMaskPadding) {
    auto cropMaskList = std::make_shared<std::list<cv::Mat>>();
    if (Globals::faceMaskerTypeSet.contains(Globals::enumFaceMaskerType::FM_Box)) {
        auto boxMask = FaceMasker::createStaticBoxMask(cropSize,
                                                       Globals::faceMaskBlur, Globals::faceMaskPadding);
        cropMaskList->push_back(*boxMask);
    } else if (Globals::faceMaskerTypeSet.contains(Globals::enumFaceMaskerType::FM_Occlusion)) {
        // todo
    } else if (Globals::faceMaskerTypeSet.contains(Globals::enumFaceMaskerType::FM_Region)) {
        // todo
    }
    return cropMaskList;
}
} // namespace Ffc