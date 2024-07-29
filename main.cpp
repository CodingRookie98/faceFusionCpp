#include "core.h"
#include "downloader.h"
#include "file_system.h"

int main() {
    std::shared_ptr<Ffc::Core> core = std::make_shared<Ffc::Core>();
    core->run();

//    auto finalizePath = Ffc::FileSystem::listFilesInDirectory("D:/0_workSpace/facefusioncppTest/finalize");
//    std::vector<std::string> finalizeFiles;
//    for (auto& file : finalizePath) {
//        finalizeFiles.push_back(file);
//    }
//    std::vector<std::string> sourceFiles;
//    for (auto& file : finalizeFiles) {
//        sourceFiles.push_back(Ffc::FileSystem::resolveRelativePath("./temp") + "/" + Ffc::FileSystem::getFileName(file));
//    }
//
//    Ffc::FileSystem::copyFiles(finalizeFiles, sourceFiles);
//    std::vector<std::string> sourceFiles2 = sourceFiles;
//    Ffc::FileSystem::finalizeImages(sourceFiles, sourceFiles2);

    return 0;
}
