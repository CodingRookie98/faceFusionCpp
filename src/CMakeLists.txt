
target_sources(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/typing.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_helper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vision.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vision.h
        ${CMAKE_CURRENT_SOURCE_DIR}/globals.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_masker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_masker.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ort_session.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ort_session.h
        ${CMAKE_CURRENT_SOURCE_DIR}/url_downloader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/url_downloader.h
)

set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2")
target_include_directories(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDA_DIR}/include
)

add_subdirectory(face_analyser)
add_subdirectory(processors)