/**
 ******************************************************************************
 * @file           : file_system.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-15
 ******************************************************************************
 */

#include <future>
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

std::vector<std::string> FileSystem::normalizeOutputPaths(const std::vector<std::string> &targetPaths, const std::string &outputPath) {
    dp::thread_pool pool(std::thread::hardware_concurrency());
    std::vector<std::future<std::string>> futures;
    for (const auto &targetPath : targetPaths) {
        futures.emplace_back(pool.enqueue([targetPath, outputPath] {
            return normalizeOutputPath(targetPath, outputPath);
        }));
    }
    std::vector<std::string> normedPaths;
    for (auto &future : futures) {
        normedPaths.push_back(future.get());
    }
    return normedPaths;
}

bool FileSystem::directoryExists(const std::string &path) {
    return std::filesystem::exists(path) && FileSystem::isDirectory(path);
}

void FileSystem::createDirectory(const std::string &path) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    if (!directoryExists(path)) {
        std::error_code ec;
        if(!std::filesystem::create_directories(path, ec)){
            std::cerr << __FUNCTION__ << " Failed to create directory: " + path + " Error: " + ec.message() << std::endl;
        }
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

std::string FileSystem::getExtension(const std::string &filePath) {
    std::filesystem::path pathObj(filePath);
    return pathObj.extension().string();
}

std::string FileSystem::getBaseName(const std::string &filePath) {
    std::filesystem::path pathObj(filePath);
    return pathObj.stem().string();
}

bool FileSystem::copyImage(const std::string &imagePath, const std::string &destination, const cv::Size &size) {
    // 读取输入图片
    cv::Mat inputImage = cv::imread(imagePath, cv::IMREAD_UNCHANGED);
    if (inputImage.empty()) {
        std::cerr << "Could not open or find the image: " << imagePath << std::endl;
        return false;
    }

    // 获取临时文件路径
    std::filesystem::path destinationPath = destination;
    if (!directoryExists(destinationPath.parent_path().string())) {
        createDirectory(destinationPath.parent_path().string());
    }

    // 调整图片尺寸
    cv::Mat resizedImage;
    cv::Size outputSize = Vision::restrictResolution(inputImage.size(), size);
    if (outputSize.width == 0 || outputSize.height == 0) {
        outputSize = inputImage.size();
    }

    if (outputSize.width != inputImage.size().width || outputSize.height != inputImage.size().height) {
        cv::resize(inputImage, resizedImage, outputSize);
    } else {
        if (destinationPath.extension() != ".webp") {
            copyFile(imagePath, destinationPath.string());
            return true;
        }
        resizedImage = inputImage;
    }

    if (destinationPath.extension() == ".webp") {
        // 设置保存参数，默认无压缩
        std::vector<int> compressionParams;
        compressionParams.push_back(cv::IMWRITE_WEBP_QUALITY);
        compressionParams.push_back(100); // 设置WebP压缩质量
        if (!cv::imwrite(destinationPath.string(), resizedImage, compressionParams)) {
            return false;
        }
    }

    return true;
}

bool FileSystem::copyImages(const std::vector<std::string> &imagePaths, const std::vector<std::string> &destinations, const cv::Size &size) {
    if (imagePaths.size() != destinations.size()) {
        std::cerr << __FUNCTION__ << " The number of image paths and destinations must be equal." << std::endl;
        return false;
    }
    if (imagePaths.empty() || destinations.empty()) {
        std::cerr << __FUNCTION__ << " No image paths or destination paths provided." << std::endl;
        return false;
    }
    
    // use multi-thread
    dp::thread_pool pool(std::thread::hardware_concurrency());

    std::vector<std::future<bool>> futures;
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        std::string imagePath = imagePaths[i];
        std::string destination = destinations[i];
        futures.emplace_back(pool.enqueue([imagePath, destination, size]() {
            return copyImage(imagePath, destination, size);
        }));
    }
    for (auto &future : futures) {
        if (!future.get()) {
            return false;
        }
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
    } else {
        outputSize = size;
    }

    if (outputSize.width != inputImage.size().width || outputSize.height != inputImage.size().height) {
        cv::resize(inputImage, resizedImage, outputSize);
    } else {
        if (outputImageQuality == 100) {
            copyFile(imagePath, outputPath);
            return true;
        }
        resizedImage = inputImage;
    }

    // 设置保存参数，默认无压缩
    std::vector<int> compressionParams;
    std::string extension = std::filesystem::path(outputPath).extension().string();

    if (extension == ".webp") {
        compressionParams.push_back(cv::IMWRITE_WEBP_QUALITY);
        compressionParams.push_back(std::clamp(outputImageQuality, 1, 100));
    } else if (extension == ".jpg" || extension == ".jpeg") {
        compressionParams.push_back(cv::IMWRITE_JPEG_QUALITY);
        compressionParams.push_back(std::clamp(outputImageQuality, 0, 100));
    } else if (extension == ".png") {
        compressionParams.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compressionParams.push_back(std::clamp((outputImageQuality / 10), 0, 9));
    }

    // 保存调整后的图像
    if (!cv::imwrite(outputPath, resizedImage, compressionParams)) {
        return false;
    }

    return true;
}

bool FileSystem::finalizeImages(const std::vector<std::string> &imagePaths, const std::vector<std::string> &outputPaths, const cv::Size &size, const int &outputImageQuality) {
    if (imagePaths.size() != outputPaths.size()) {
        throw std::invalid_argument("Input and output paths must have the same size");
    }

    dp::thread_pool pool(std::thread::hardware_concurrency());
    std::vector<std::future<bool>> futures;
    for (size_t i = 0; i < imagePaths.size(); ++i) {
        futures.emplace_back(pool.enqueue([imagePath = imagePaths[i], outputPath = outputPaths[i], size, outputImageQuality]() {
            //            return finalizeImage(imagePath, outputPath, size, outputImageQuality);
            try {
                return finalizeImage(imagePath, outputPath, size, outputImageQuality);
            } catch (const std::exception &e) {
                // 记录异常或处理异常
                std::cerr << "Exception caught: " << e.what() << std::endl;
                return false; // 返回错误状态
            } catch (...) {
                // 捕获所有其他异常
                std::cerr << "Unknown exception caught" << std::endl;
                return false; // 返回错误状态
            }
        }));
    }

    bool allSuccess = true;
    for (auto &future : futures) {
        bool success = future.get();
        if (!success) {
            allSuccess = false;
        }
    }

    return allSuccess;
}

void FileSystem::removeDirectory(const std::string &path) {
    try {
        std::filesystem::remove_all(path);
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << __FUNCTION__ << " Error: " << e.what() << std::endl;
    }
}

void FileSystem::removeFile(const std::string &path) {
    try {
        std::filesystem::remove(path);
    } catch (const std::filesystem::filesystem_error &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void FileSystem::copyFile(const std::string &source, const std::string &destination) {
    if(source == destination) return;
    if (!fileExists(source)) {
        throw std::invalid_argument("Source file does not exist");
    }
    
    // parent path of destination is not exist
    if (!isDirectory(std::filesystem::path(destination).parent_path().string())) {
        createDirectory(std::filesystem::path(destination).parent_path().string());
    }
    
    std::filesystem::copy(source, destination, std::filesystem::copy_options::overwrite_existing);
}

void FileSystem::copyFiles(const std::vector<std::string> &sources, const std::vector<std::string> &destination) {
    if (sources.size() != destination.size()) {
        throw std::invalid_argument("Source and destination paths must have the same size");
    }

    dp::thread_pool pool(std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < sources.size(); ++i) {
        futures.emplace_back(pool.enqueue([sources, destination, i]() {
            copyFile(sources[i], destination[i]);
        }));
    }
    for (auto &future : futures) {
        future.get();
    }
}

void FileSystem::moveFile(const std::string &source, const std::string &destination) {
    // if parent path of destination is not exist
    if (!isDirectory(std::filesystem::path(destination).parent_path().string())) {
        createDirectory(std::filesystem::path(destination).parent_path().string());
    }
    
    // if destination is existed
    if (fileExists(destination)) {
        removeFile(destination);
    }
    
    std::filesystem::rename(source, destination);
}

void FileSystem::moveFiles(const std::vector<std::string> &sources,
                           const std::vector<std::string> &destination) {
    if (sources.size() != destination.size()) {
        throw std::invalid_argument("Source and destination paths must have the same size");
    }

    dp::thread_pool pool(std::thread::hardware_concurrency());
    std::vector<std::future<void>> futures;
    for (size_t i = 0; i < sources.size(); ++i) {
        futures.emplace_back(pool.enqueue([sources, destination, i]() {
            moveFile(sources[i], destination[i]);
        }));
    }
    for (auto &future : futures) {
        future.get();
    }
}

std::string FileSystem::generateRandomString(const size_t &length) {
    const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<size_t> distribution(0, characters.size() - 1);

    std::string randomString;
    for (size_t i = 0; i < length; ++i) {
        randomString += characters[distribution(generator)];
    }

    return randomString;
}
} // namespace Ffc