/**
 ******************************************************************************
 * @file           : file_system.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-15
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_FILE_SYSTEM_H_
#define FACEFUSIONCPP_SRC_FILE_SYSTEM_H_

#include <filesystem>
#include <string>
#include <opencv2/opencv.hpp>
#include <unordered_set>
#include <vector>
#include <random>

namespace Ffc {

class FileSystem {
public:
    FileSystem() = default;
    ~FileSystem() = default;
    static bool fileExists(const std::string &path);
    static bool isDirectory(const std::string &path);
    static bool isFile(const std::string &path);
    static bool isImage(const std::string &path);
    static std::string getFileNameFromURL(const std::string &url);
    static uintmax_t getFileSize(const std::string &path);
    static std::unordered_set<std::string> listFilesInDirectory(const std::string &path);
    static std::string resolveRelativePath(const std::string &path);
    static bool hasImage(const std::unordered_set<std::string> &paths);
    static std::unordered_set<std::string> filterImagePaths(const std::unordered_set<std::string> &paths);
    static std::string normalizeOutputPath(const std::string &targetPath, const std::string &outputPath);
    static bool directoryExists(const std::string &path);
    static void createDirectory(const std::string &path);
    static void removeDirectory(const std::string &path);
    static void removeFile(const std::string &path);
    static void copyFile(const std::string &source, const std::string &destination);
    static void moveFile(const std::string &source, const std::string &destination);
    static std::string getTempPath();
    static std::string getFileName(const std::string &filePath);
    static bool copyImageToTemp(const std::string &imagePath, const cv::Size &size = cv::Size(0, 0));
    static bool finalizeImage(const std::string &imagePath, const std::string &outputPath, const cv::Size &size = cv::Size(0, 0), const int &outputImageQuality = 100);
    static std::string generateRandomString(const size_t &length);
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_FILE_SYSTEM_H_
