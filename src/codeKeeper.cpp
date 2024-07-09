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

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModelLoader");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Load the ONNX model
    const char *model_path = "path/to/your/model.onnx";
    Ort::Session session(env, model_path, session_options);

    // Load ONNX model as a protobuf message
    onnx::ModelProto model_proto;
    std::ifstream input(model_path, std::ios::binary);
    if (!model_proto.ParseFromIstream(&input)) {
        std::cerr << "Failed to load model." << std::endl;
        return -1;
    }

    // Access the initializer
    const onnx::TensorProto &initializer = model_proto.graph().initializer(model_proto.graph().initializer_size() - 1);

    // Convert initializer to an array
    std::vector<float> initializer_array(initializer.float_data().begin(), initializer.float_data().end());

    // Print the array (for example purposes)
    for (float value : initializer_array) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
