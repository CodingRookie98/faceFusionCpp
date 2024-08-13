/**
 ******************************************************************************
 * @file           : processor_base.h
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-20
 ******************************************************************************
 */

#ifndef FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_PROCESSOR_BASE_H_
#define FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_PROCESSOR_BASE_H_

#include <unordered_set>
#include <string>
#include "typing.h"

namespace Ffc {

class ProcessorBase {
protected:
public:
    ProcessorBase() = default;
    virtual ~ProcessorBase() = default;

    virtual bool preCheck() = 0;
    virtual bool postCheck() = 0;
    virtual bool preProcess() = 0;
    virtual Typing::VisionFrame getReferenceFrame(const Typing::Face &sourceFace,
                                                  const Typing::Face &targetFace,
                                                  const Typing::VisionFrame &tempVisionFrame) = 0;
    virtual void processImage(const std::unordered_set<std::string> &sourcePaths,
                              const std::string &targetPath,
                              const std::string &outputPath) = 0;
    virtual void processImages(const std::unordered_set<std::string> &sourcePaths,
                               const std::vector<std::string> &targetPaths,
                               const std::vector<std::string> &outputPaths) = 0;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_PROCESSOR_BASE_H_
