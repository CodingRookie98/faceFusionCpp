#include "core.h"
#include "downloader.h"

int main() {
    std::shared_ptr<Ffc::Core> core = std::make_shared<Ffc::Core>();
    core->run();

    //    bool downloadFlag = Ffc::Downloader::download("https://github.com/facefusion/facefusion-assets/releases/download/models/2dfan4.onnx", "./temp");
//    if (downloadFlag) {
//        std::cout << "Download success" << std::endl;
//    } else {
//        std::cout << "Download failed" << std::endl;
//    }

//    system("pause");

    return 0;
}
