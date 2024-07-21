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

namespace Ffc {

class ProcessorBase {
public:
    ProcessorBase() = default;
    virtual ~ProcessorBase() = default;

    virtual bool preCheck() = 0;
    virtual bool postCheck() = 0;
    virtual bool preProcess() = 0;
};

} // namespace Ffc

#endif // FACEFUSIONCPP_SRC_PROCESSORS_FRAME_MODULES_PROCESSOR_BASE_H_
