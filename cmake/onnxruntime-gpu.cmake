# 设置下载和解压的路径
set(DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set(EXTRACT_DIR "${CMAKE_BINARY_DIR}")

# 下载文件的 URL 和目标文件名
set(DOWNLOAD_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-win-x64-gpu-cuda12-1.18.1.zip")
set(DOWNLOAD_FILE "${DOWNLOAD_DIR}/onnxruntime-win-x64-gpu-cuda12-1.18.1.zip")

# 检查是否已经下载并解压
if(NOT EXISTS "${EXTRACT_DIR}/downloads/onnxruntime-win-x64-gpu-cuda12-1.18.1.zip")
    # 确保下载目录存在
    file(MAKE_DIRECTORY "${DOWNLOAD_DIR}")

    # 下载文件
    message(STATUS "Downloading onnxruntime-win-x64-gpu...")
    file(DOWNLOAD "${DOWNLOAD_URL}" "${DOWNLOAD_FILE}" SHOW_PROGRESS)

    # 确保解压目录存在
    file(MAKE_DIRECTORY "${EXTRACT_DIR}")
else()
    message(STATUS "onnxruntime-win-x64-gpu has already been downloaded.")
endif()

if (NOT EXISTS "${CMAKE_BINARY_DIR}/onnxruntime-win-x64-gpu-1.18.1/lib/onnxruntime.dll")
    # 解压文件
    message(STATUS "Extracting onnxruntime-win-x64-gpu...")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${DOWNLOAD_FILE}"
        WORKING_DIRECTORY "${EXTRACT_DIR}"
    )
else ()
    message(STATUS "onnxruntime-win-x64-gpu has already been extracted.")
endif()

FILE(GLOB ONNXRUNTIME_DLLS "${CMAKE_BINARY_DIR}/onnxruntime-win-x64-gpu-1.18.1/lib/*.dll")

add_custom_command(TARGET ${ProgramName} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ONNXRUNTIME_DLLS}
        $<TARGET_FILE_DIR:${ProgramName}>
    COMMENT "Copying ONNX Runtime DLLs to runtime output directory"
)