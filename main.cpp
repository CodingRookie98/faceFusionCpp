#include "core.h"

int main() {
    std::shared_ptr<Ffc::Core> core = std::make_shared<Ffc::Core>();
    core->run();

    return 0;
}
