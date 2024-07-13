/**
 ******************************************************************************
 * @file           : url_downloader.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-13
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_URL_DOWNLOADER_H_
#define FACEFUSIONCPP_SRC_URL_DOWNLOADER_H_

#include <string>
#include <iostream>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
// windows
#include <windows.h>
#else
// POSIX
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#endif

namespace Ffc {

class UrlDownloader {
public:
    static bool downloadFile(const std::string &url, const std::string &outPutDirectory);

private:
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_URL_DOWNLOADER_H_
