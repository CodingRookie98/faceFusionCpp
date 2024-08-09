cmake_minimum_required(VERSION 3.28)
project(faceFusionCpp VERSION 0.1.0 DESCRIPTION "This project is the C++ version of the open-source project FaceFusion.")

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/bin/Debug)
Set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/bin/Release)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif ()

set(ProgramName ${PROJECT_NAME})
add_executable(${ProgramName})

include("./cmake/onnxruntime-gpu.cmake")

add_subdirectory(facefusionCpp)
set(ONNXRUNTIME "${CMAKE_BINARY_DIR}/onnxruntime-win-x64-gpu-1.18.1")

find_package(OpenCV REQUIRED)
find_package(ONNX CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_path(SIMPLEINI_INCLUDE_DIRS "ConvertUTF.c")
find_package(spdlog CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(indicators CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(thread-pool CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS process)

target_include_directories(${ProgramName} PRIVATE ${SIMPLEINI_INCLUDE_DIRS})
target_link_directories(${ProgramName} PRIVATE ${VCPKG_INSTALLED_DIR}/x64-windows/lib)
target_link_libraries(${ProgramName} PRIVATE ${OpenCV_LIBS})
target_include_directories(${ProgramName} PRIVATE "${ONNXRUNTIME}/include")
target_link_directories(${ProgramName} PRIVATE "${ONNXRUNTIME}/lib")
target_link_libraries(${ProgramName} PRIVATE onnxruntime)
target_link_libraries(${ProgramName} PRIVATE onnx onnx_proto)
target_link_libraries(${ProgramName} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${ProgramName} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${ProgramName} PRIVATE CURL::libcurl)
target_link_libraries(${ProgramName} PRIVATE indicators::indicators)
target_link_libraries(${ProgramName} PRIVATE OpenSSL::Crypto)
target_link_libraries(${ProgramName} PRIVATE dp::thread-pool)
target_link_libraries(${ProgramName} PRIVATE Boost::process)

list(APPEND RESOURCE_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/faceFusionCpp.ini"
        "${CMAKE_CURRENT_SOURCE_DIR}/facefusionCpp/modelsInfo.json"
)

add_custom_target(COPY_RESOURCE_FILES
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${RESOURCE_FILES}
        $<TARGET_FILE_DIR:${ProgramName}>
        COMMENT "Copying resource files to runtime output directory"
)
add_dependencies(${ProgramName} COPY_RESOURCE_FILES)

# 设置打包的方式
set(CPACK_GENERATOR ZIP)

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${faceFusionCpp_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${faceFusionCpp_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${faceFusionCpp_VERSION_PATCH}")
set(CPACK_PACKAGE_NAME ${PROJECT_NAME})

if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-win-x64-${${PROJECT_NAME}_VERSION}")
elseif (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-linux-x64-${${PROJECT_NAME}_VERSION}")
else ()
    message(STATUS "The system is neither Windows nor Linux.")
endif ()

set(INSTALL_PATH "./")
set(X_VCPKG_APPLOCAL_DEPS_INSTALL TRUE CACHE BOOL "Automatically COPY dependencies into the install target directory for executables.")
set(X_VCPKG_APPLOCAL_DEPS_SERIALIZED ON)

install(TARGETS ${ProgramName} DESTINATION ${INSTALL_PATH}
        RUNTIME LIBRARY ARCHIVE COMPONENT "app_exe" DESTINATION ${INSTALL_PATH}
)

install(FILES $<TARGET_RUNTIME_DLLS:${ProgramName}>
        COMPONENT "app_dll"
        DESTINATION ${INSTALL_PATH}
)
install(FILES ${RESOURCE_FILES} DESTINATION ${INSTALL_PATH})
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/DOCUMENT.md"
        "${CMAKE_CURRENT_SOURCE_DIR}/DISCLAIMER.txt"
        DESTINATION ${INSTALL_PATH}
)

# Please configure cmake again after building the project
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    file(GLOB RUNTIME_FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}/*.dll")
elseif (${CMAKE_BUILD_TYPE} STREQUAL "Release")
    file(GLOB RUNTIME_FILES "${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/*.dll")
endif ()
install(FILES ${RUNTIME_FILES} DESTINATION ${INSTALL_PATH})

include(InstallRequiredSystemLibraries)
include(CPack)
