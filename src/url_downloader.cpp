/**
 ******************************************************************************
 * @file           : url_downloader.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-13
 ******************************************************************************
 */

#include "url_downloader.h"

namespace Ffc {

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
bool UrlDownloader::downloadFile(const std::string &url, const std::string &outPutDirectory) {
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    std::string command;
    command = command + "wget2" + " -P " + outPutDirectory + " " + url;

    // Convert string to LPSTR
    char *commandLine = new char[command.size() + 1];
    strcpy(commandLine, command.c_str());

    if (CreateProcess(NULL,        // No module name (use command line)
                      commandLine, // Command line
                      NULL,        // Process handle not inheritable
                      NULL,        // Thread handle not inheritable
                      FALSE,       // Set handle inheritance to FALSE
                      0,           // No creation flags
                      NULL,        // Use parent's environment block
                      NULL,        // Use parent's starting directory
                      &si,         // Pointer to STARTUPINFO structure
                      &pi)         // Pointer to PROCESS_INFORMATION structure
    ) {
        // Wait until child process exits.
        WaitForSingleObject(pi.hProcess, INFINITE);

        // Close process and thread handles.
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
    } else {
        std::cerr << "CreateProcess failed (" << GetLastError() << ").\n";
    }

    delete[] commandLine;
    return 0;
}
#else
bool UrlDownloader::downloadFile(const std::string &url, const std::string &outPutDirectory) {
    pid_t pid = fork();

    if (pid == 0) {
        // Child process
        const char *program = "./wget";
        const char *arguments[] = {program, "-P ", outPutDirectory.c_str(), " ",url.c_str(), nullptr};

        execvp(program, const_cast<char *const *>(arguments));

        // If execvp returns, it must have failed.
        std::cerr << "Exec failed\n";
        return 1;
    } else if (pid > 0) {
        // Parent process
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) {
            std::cout << "Child exited with status " << WEXITSTATUS(status) << "\n";
        }
    } else {
        // Fork failed
        std::cerr << "Fork failed\n";
        return 1;
    }
}
#endif

} // namespace Ffc