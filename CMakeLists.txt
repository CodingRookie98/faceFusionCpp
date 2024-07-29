cmake_minimum_required(VERSION 3.28)
project(faceFusionCpp)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug)
Set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release)

add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")

set(ProgramName "faceFusion")
add_executable(${ProgramName} main.cpp)

add_subdirectory(src)

find_package(OpenCV REQUIRED)
find_package(ONNX CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_path(SIMPLEINI_INCLUDE_DIRS "ConvertUTF.c")
find_package(spdlog CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(indicators CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(thread-pool CONFIG REQUIRED)

target_include_directories(${ProgramName} PRIVATE ${SIMPLEINI_INCLUDE_DIRS})
target_link_directories(${ProgramName} PRIVATE ${VCPKG_INSTALLED_DIR}/x64-windows/lib)
target_link_libraries(${ProgramName} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${ProgramName} PRIVATE onnxruntime)
target_link_libraries(${ProgramName} PRIVATE onnx onnx_proto)
target_link_libraries(${ProgramName} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${ProgramName} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${ProgramName} PRIVATE CURL::libcurl)
target_link_libraries(${ProgramName} PRIVATE indicators::indicators)
target_link_libraries(${ProgramName} PRIVATE OpenSSL::Crypto)
target_link_libraries(${ProgramName} PRIVATE dp::thread-pool)
