cmake_minimum_required(VERSION 3.28)
project(faceFusionCpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug)
Set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release)

set(ProgramName "faceFusion")
add_executable(${ProgramName} main.cpp)

add_subdirectory(src)

find_package(OpenCV REQUIRED)
find_package(ONNX CONFIG REQUIRED)

target_link_directories(${ProgramName} PRIVATE
        ${VCPKG_INSTALLED_DIR}/x64-windows/lib
)

target_link_libraries(${ProgramName} PRIVATE
        ${OpenCV_LIBS}

)
# 链接 ONNX Runtime库
target_link_libraries(${ProgramName} PRIVATE onnxruntime)

target_link_libraries(${ProgramName} PRIVATE onnx onnx_proto)