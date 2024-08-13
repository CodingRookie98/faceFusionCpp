
target_sources(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/typing.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_helper.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_helper.h
        ${CMAKE_CURRENT_SOURCE_DIR}/vision.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/vision.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_masker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_masker.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ort_session.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ort_session.h
        ${CMAKE_CURRENT_SOURCE_DIR}/downloader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/downloader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/file_system.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/file_system.h
        ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/config.h
        ${CMAKE_CURRENT_SOURCE_DIR}/core.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/core.h
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/logger.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_store.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_store.h
        ${CMAKE_CURRENT_SOURCE_DIR}/progress_bar.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/progress_bar.h
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_runner.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ffmpeg_runner.h
        ${CMAKE_CURRENT_SOURCE_DIR}/metadata.h
)

target_include_directories(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(face_analyser)
add_subdirectory(processors)