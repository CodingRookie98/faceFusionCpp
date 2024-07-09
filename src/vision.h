/**
 ******************************************************************************
 * @file           : vision.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_VISION_H_
#define FACEFUSIONCPP_SRC_VISION_H_

#include <opencv2/opencv.hpp>
#include "typing.h"

namespace Ffc {

class Vision {
public:
    static std::vector<cv::Mat> readStaticImages(const std::vector<std::string> &imagePaths);
    static cv::Mat readStaticImage(const std::string &imagePath);
    static Typing::VisionFrame
    resizeFrameResolution(const Typing::VisionFrame &visionFrame, const cv::Size &cropSize);
    static bool writeImage(const cv::Mat &image, const std::string &imagePath);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_VISION_H_
