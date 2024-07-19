/**
 ******************************************************************************
 * @file           : file_system.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-15
 ******************************************************************************
 */

#include "file_system.h"

namespace Ffc {

bool FileSystem::fileExists(const std::string &path) {
    return std::filesystem::exists(path);
}

bool FileSystem::isDirectory(const std::string &path) {
    return std::filesystem::is_directory(path);
}

bool FileSystem::isFile(const std::string &path) {
    return std::filesystem::is_regular_file(path);
}

bool FileSystem::isImage(const std::string &path) {
    if (!isFile(path)) {
        return false;
    } else {
        return cv::haveImageReader(path);
    }
}

std::string FileSystem::getFileNameFromURL(const std::string &url) {
    std::size_t lastSlashPos = url.find_last_of('/');
    if (lastSlashPos == std::string::npos) {
        return url;
    }

    std::string fileName = url.substr(lastSlashPos + 1);
    return fileName;
}

uintmax_t FileSystem::getFileSize(const std::string &path) {
    if (isFile(path)) {
        return std::filesystem::file_size(path);
    } else {
        return 0;
    }
}

std::unordered_set<std::string> FileSystem::listFilesInDirectory(const std::string &path) {
    std::unordered_set<std::string> filePaths;

    if (!isDirectory(path)) {
        throw std::invalid_argument("Path is not a directory");
    }

    try {
        for (const auto &entry : std::filesystem::directory_iterator(path)) {
            if (std::filesystem::is_regular_file(entry.status())) {
                filePaths.insert(std::filesystem::absolute(entry.path()).string());
            }
        }
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return filePaths;
}

std::vector<std::string> FileSystem::filterImagePaths(const std::vector<std::string> &paths) {
    std::vector<std::string> imagePaths;

    for (const auto &path : paths) {
        if (isImage(path)) {
            imagePaths.push_back(path);
        }
    }

    return imagePaths;
}
} // namespace Ffc