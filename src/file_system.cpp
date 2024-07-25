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

std::string FileSystem::resolveRelativePath(const std::string &path) {
    return std::filesystem::absolute(path).string();
}

bool FileSystem::hasImage(const std::unordered_set<std::string> &paths) {
    std::unordered_set<std::string> imagePaths;
    for (auto &path : paths) {
        auto absPath = resolveRelativePath(path);
        if (!isImage(absPath)) {
            return false;
        }
    }
    return true;
}

std::unordered_set<std::string> FileSystem::filterImagePaths(const std::unordered_set<std::string> &paths) {
    std::unordered_set<std::string> imagePaths;
    for (auto &path : paths) {
        auto absPath = resolveRelativePath(path);
        if (isImage(absPath)) {
            imagePaths.insert(absPath);
        }
    }
    return imagePaths;
}

std::string FileSystem::normalizeOutputPath(const std::string &targetPath, const std::string &outputPath) {
    if (!targetPath.empty() && !outputPath.empty()) {
        std::filesystem::path targetPathObj(targetPath);
        std::filesystem::path outputPathObj(outputPath);
        std::string targetFileName = targetPathObj.stem().string();
        std::string targetFileExtension = targetPathObj.extension().string();

        if (isDirectory(outputPath)) {
            std::string suffix = generateRandomString(8);
            std::string normedPath = resolveRelativePath(outputPath + "/" + targetFileName + "-" + suffix + targetFileExtension);
            while (fileExists(normedPath)) {
                suffix = generateRandomString(8);
                normedPath = resolveRelativePath(outputPath + "/" + targetFileName + "-" + suffix + targetFileExtension);
            }
            return normedPath;
        } else {
            std::string outputDir = outputPathObj.parent_path().string();
            std::string outputFileName = outputPathObj.stem().string();
            std::string outputExtension = outputPathObj.extension().string();
            if (isDirectory(outputDir) && !outputFileName.empty() && !outputExtension.empty()) {
                return resolveRelativePath(outputDir + "/" + outputFileName + outputExtension);
            }
        }
    }
    return {};
}

bool FileSystem::directoryExists(const std::string &path) {
    return std::filesystem::exists(path) && FileSystem::isDirectory(path);
}

void FileSystem::createDirectory(const std::string &path) {
    if (!directoryExists(path)) {
        std::filesystem::create_directory(path);
    }
}

std::string FileSystem::getTempPath() {
    std::string tempPath = resolveRelativePath("./temp");
    createDirectory(tempPath);
    return tempPath;
}

std::string FileSystem::getFileName(const std::string &filePath) {
    std::filesystem::path pathObj(filePath);
    return pathObj.filename().string();
}

bool FileSystem::copyImageToTemp(const std::string &imagePath, const cv::Size &size) {
    // 读取输入图片
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image: " << imagePath << std::endl;
        return false;
    }

    // 调整图片尺寸
    cv::Mat resizedImage;
    cv::Size outputSize;
    if (size.width == 0 || size.height == 0) {
        outputSize = inputImage.size();
    }
    cv::resize(inputImage, resizedImage, outputSize);

    // 获取临时文件路径
    std::filesystem::path tempFilePath = getTempPath() + "/" + getFileName(imagePath);
    tempFilePath.replace_extension(std::filesystem::path(imagePath).extension());

    // 设置保存参数，默认无压缩
    std::vector<int> compressionParams;
    if (tempFilePath.extension() == ".webp") {
        compressionParams.push_back(cv::IMWRITE_WEBP_QUALITY);
        compressionParams.push_back(100); // 设置WebP压缩质量
    }

    if (!cv::imwrite(tempFilePath.string(), resizedImage, compressionParams)) {
        return false;
    }

    return true;
}

bool FileSystem::finalizeImage(const std::string &imagePath, const std::string &outputPath, const cv::Size &size, const int &outputImageQuality) {
    // 读取输入图像
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        return false;
    }

    // 调整图像大小
    cv::Mat resizedImage;
    cv::Size outputSize;
    if (size.width == 0 || size.height == 0) {
        outputSize = inputImage.size();
    }
    cv::resize(inputImage, resizedImage, outputSize);

    // 设置保存参数，默认无压缩
    std::vector<int> compression_params;
    if (std::filesystem::path(outputPath).extension() == ".webp") {
        compression_params.push_back(cv::IMWRITE_WEBP_QUALITY);
        compression_params.push_back(outputImageQuality);
    } else if (std::filesystem::path(outputPath).extension() == ".jpg"
               || std::filesystem::path(outputPath).extension() == ".jpeg") {
        compression_params.push_back(cv::IMWRITE_JPEG_QUALITY);
        compression_params.push_back(outputImageQuality);
    } else if (std::filesystem::path(outputPath).extension() == ".png") {
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(outputImageQuality);
    }

    // 保存调整后的图像
    if (!cv::imwrite(outputPath, resizedImage, compression_params)) {
        //        std::cerr << "Could not write the image to: " << outputPath << std::endl;
        return false;
    }

    return true;
}
void FileSystem::removeDirectory(const std::string &path) {
    std::filesystem::remove_all(path);
}

void FileSystem::removeFile(const std::string &path) {
    std::filesystem::remove(path);
}

void FileSystem::copyFile(const std::string &source, const std::string &destination) {
    std::filesystem::copy(source, destination, std::filesystem::copy_options::overwrite_existing);
}

void FileSystem::moveFile(const std::string &source, const std::string &destination) {
    std::filesystem::rename(source, destination);
}

std::string FileSystem::generateRandomString(const size_t &length) {
    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::default_random_engine generator(static_cast<unsigned long>(std::time(0)));
    std::uniform_int_distribution<size_t> distribution(0, characters.size() - 1);

    std::string randomString;
    for (size_t i = 0; i < length; ++i) {
        randomString += characters[distribution(generator)];
    }

    return randomString;
}
} // namespace Ffc