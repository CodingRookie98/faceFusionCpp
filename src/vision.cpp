/**
 ******************************************************************************
 * @file           : vision.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-6
 ******************************************************************************
 */

#include "vision.h"

namespace Ffc {
std::vector<cv::Mat> Vision::readStaticImages(const std::vector<std::string> &imagePaths) {
    std::vector<cv::Mat> images;
    for (const auto &imagePath : imagePaths) {
        images.emplace_back(readStaticImage(imagePath));
    }

    return images;
}

cv::Mat Vision::readStaticImage(const std::string &imagePath) {
    //    BGR
    return cv::imread(imagePath, cv::IMREAD_COLOR);
}

Typing::VisionFrame Vision::resizeFrameResolution(const Typing::VisionFrame &visionFrame, const cv::Size &cropSize) {
    const int height = visionFrame.rows;
    const int width = visionFrame.cols;
    cv::Mat tempImage = visionFrame.clone();
    if (height > cropSize.height || width > cropSize.width) {
        const float scale = std::min((float)cropSize.height / height, (float)cropSize.width / width);
        cv::Size newSize = cv::Size(int(width * scale), int(height * scale));
        cv::resize(visionFrame, tempImage, newSize);
    }
    return tempImage;
}

bool Vision::writeImage(const cv::Mat &image, const std::string &imagePath) {
    // You may encounter long path problems in Windows
    if (cv::imwrite(imagePath, image)) {
        return true;
    }
    return false;
}
} // namespace Ffc