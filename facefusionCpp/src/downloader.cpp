/**
 ******************************************************************************
 * @file           : downloader.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-13
 ******************************************************************************
 */

#include "downloader.h"

namespace Ffc {

size_t Downloader::writeData(void *ptr, size_t size, size_t nmemb, std::ofstream *stream) {
    stream->write(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
}

long Downloader::getFileSize(const std::string &url) {
    CURL *curl;
    CURLcode res;
    double fileSize = 0.0;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); // 不下载内容
        curl_easy_setopt(curl, CURLOPT_HEADER, 1L); // 只获取头部信息

        // 忽略响应头
        //        curl_easy_setopt(curl, CURLOPT_HEADER, 0L);
        // 关闭详细调试信息
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
        // 跟随重定向
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // 设置SSL选项
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L); // 验证服务器的SSL证书
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L); // 验证证书上的主机名

        // 自定义处理头部数据
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, headerCallback);

        res = curl_easy_perform(curl);

        if (res == CURLE_OK) {
            res = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &fileSize);
            if ((res == CURLE_OK) && (fileSize > 0.0)) {
                curl_easy_cleanup(curl);
                return static_cast<long>(fileSize);
            }
        } else {
            Logger::getInstance()->error(std::format("curl_easy_perform() failed: {}", curl_easy_strerror(res)));
        }

        curl_easy_cleanup(curl);
    }

    return -1;
}

size_t Downloader::headerCallback(char *buffer, size_t size, size_t nitems, void *userdata) {
    return size * nitems;
}

bool Downloader::isDownloadDone(const std::string &Url, const std::string &filePath) {
    if (FileSystem::fileExists(filePath) && FileSystem::getFileSize(filePath) == getFileSize(Url)) {
        return true;
    }
    return false;
}

bool Downloader::batchDownload(const std::unordered_set<std::string> &urls, const std::string &outPutDirectory) {
    std::string outputDir = FileSystem::resolveRelativePath(outPutDirectory);
    for (const auto &url : urls) {
        if (!download(url, outputDir)) {
            return false;
        }
    }
    return true;
}

bool Downloader::download(const std::string &url, const std::string &outPutDirectory) {
    std::string outputDir = FileSystem::resolveRelativePath(outPutDirectory);
    if (!std::filesystem::exists(outputDir)) {
        if (!std::filesystem::create_directories(outputDir)) {
            Logger::getInstance()->error(std::format("Failed to create output directory: {}", outputDir));
            return false;
        }
    }

    //  获取文件大小
    long fileSize = getFileSize(url);
    if (fileSize == -1) {
        Logger::getInstance()->error("Failed to get file size.");
        return false;
    }

    // 获取文件路径
    std::string outputFilePath = FileSystem::resolveRelativePath(outputDir + "/" + getFileNameFromUrl(url));
    std::ofstream output(outputFilePath, std::ios::binary);
    if (!output.is_open()) {
        Logger::getInstance()->error(std::format("Failed to open output file: {}", outputFilePath));
        return false;
    }

    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();
    if (curl) {
        // 设置URL
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // 设置Range头
        //        std::string range = std::to_string(start) + "-" + std::to_string(end);
        //        curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
        // 忽略响应头
        curl_easy_setopt(curl, CURLOPT_HEADER, 0L);
        // 关闭详细调试信息
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
        // 设置写入数据的回调函数
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeData);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &output);

        // 设置进度回调函数
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);

        ProgressBar *bar = new ProgressBar;
        bar->setMaxProgress(100);
        bar->setProgress(0);
        bar->setPrefixText("Downloading " + getFileNameFromUrl(url) + ": ");
        bar->setPostfixText(std::format("{}/{}", humanReadableSize(0), humanReadableSize(fileSize)));
        // 传递给回调函数的自定义数据
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, bar);

        // 自定义处理头部数据
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, headerCallback);

        // 启用进度回调 1L为禁用
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);

        // 跟随重定向
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

        // 设置SSL选项
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L); // 验证服务器的SSL证书
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L); // 验证证书上的主机名

        bar->showConsoleCursor(false);
        // 执行请求
        res = curl_easy_perform(curl);
        bar->showConsoleCursor(true);

        // 清理
        curl_easy_cleanup(curl);
        delete bar;

        if (res != CURLE_OK) {
            Logger::getInstance()->error(std::format("curl_easy_perform() failed: {}", curl_easy_strerror(res)));
            return false;
        }
        return true;
    }
    if (FileSystem::fileExists(outputFilePath) && FileSystem::getFileSize(outputFilePath) == fileSize) {
        Logger::getInstance()->info("File already exists.");
        return true;
    } else {
        Logger::getInstance()->error("File downloaded failed.");
        return false;
    }
}

std::string Downloader::getFileNameFromUrl(const std::string &url) {
    std::string::size_type pos = url.find_last_of("/\\");
    if (pos != std::string::npos) {
        return url.substr(pos + 1);
    }
    return {};
}

int Downloader::progressCallback(void *clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    // dltotal: 总下载数据量
    // dlnow: 当前已下载的数据量
    // ultotal: 总上传数据量（对上传请求有用）
    // ulnow: 当前已上传的数据量（对上传请求有用）

    // 获取进度条对象
    ProgressBar *bar = static_cast<ProgressBar *>(clientp);
    bar->setPostfixText(std::format("{}/{}", humanReadableSize(dlnow), humanReadableSize(dltotal)));
    bar->setProgress(std::round((float)dlnow / (float)dltotal * 100.0f));

    // 返回0继续操作，返回1中止操作
    return 0;
}

std::string Downloader::humanReadableSize(long size) {
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int i = 0;
    double size_d = static_cast<double>(size);

    while (size_d >= 1024.0) {
        size_d /= 1024.0;
        i++;
    }
    return std::format("{:.2f} {}", size_d, units[i]);
}
} // namespace Ffc