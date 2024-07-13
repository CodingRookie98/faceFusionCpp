/**
 ******************************************************************************
 * @file           : codeKeeper.cpp
 * @author         : CodingRookie
 * @brief          : None
 * @attention      : None
 * @date           : 24-7-7
 ******************************************************************************
 */

// model = onnx.load(model_path)
//			MODEL_INITIALIZER = numpy_helper.to_array(model.graph.initializer[-1])
//			python代码转为c++
#include <iostream>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#include <onnx/onnx_pb.h>
#include <onnxruntime/core/graph/onnx_protobuf.h>

cv::Mat computeMinMask(const std::vector<cv::Mat>& masks) {
    if (masks.empty()) {
        throw std::invalid_argument("The input vector is empty.");
    }
    
    // Initialize the minMask with the first mask in the vector
    cv::Mat minMask = masks[0].clone();
    
    for (size_t i = 1; i < masks.size(); ++i) {
        // Check if the current mask has the same size and type as minMask
        if (masks[i].size() != minMask.size() || masks[i].type() != minMask.type()) {
            throw std::invalid_argument("All masks must have the same size and type.");
        }
        cv::min(minMask, masks[i], minMask);
    }
    
    return minMask;
}

