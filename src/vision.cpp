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
        cv::Mat image = readStaticImage(imagePath);
        if (!image.empty()) {
            images.emplace_back(image);
        }
    }

    return images;
}

cv::Mat Vision::readStaticImage(const std::string &imagePath) {
    //    BGR
    if (FileSystem::fileExists(imagePath) && FileSystem::isFile(imagePath) && FileSystem::isImage(imagePath)) {
        return cv::imread(imagePath, cv::IMREAD_COLOR);
    } else if (!FileSystem::fileExists(imagePath)) {
        throw std::invalid_argument("File does not exist: " + imagePath);
    } else if (!FileSystem::isFile(imagePath)) {
        throw std::invalid_argument("Path is not a file: " + imagePath);
    } else if (!FileSystem::isImage(imagePath)) {
        throw std::invalid_argument("Path is not an image file: " + imagePath);
    } else {
        throw std::invalid_argument("Unknown error occurred while reading image: " + imagePath);
    }
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
    if (image.empty()) {
        return false;
    }
    // You may encounter long path problems in Windows
    if (cv::imwrite(imagePath, image)) {
        return true;
    }
    return false;
}

cv::Size Vision::unpackResolution(const std::string &resolution) {
    int width = 0;
    int height = 0;
    char delimiter = 'x';

    std::stringstream ss(resolution);
    ss >> width >> delimiter >> height;

    if (ss.fail()) {
        throw std::invalid_argument("Invalid dimensions format");
    }

    return {width, height};
}

std::vector<cv::Mat> Vision::readStaticImages(const std::unordered_set<std::string> &imagePaths) {
    std::vector<cv::Mat> images;
    for (const auto &imagePath : imagePaths) {
        cv::Mat image = readStaticImage(imagePath);
        if (!image.empty()) {
            images.emplace_back(image);
        }
    }

    return images;
}

cv::Size Vision::restrictResolution(const cv::Size &resolution1, const cv::Size &resolution2) {
    uint64_t area1 = static_cast<uint64_t>(resolution1.width) * resolution1.height;
    uint64_t area2 = static_cast<uint64_t>(resolution2.width) * resolution2.height;
    return area1 < area2 ? resolution1 : resolution2;
}
} // namespace Ffc