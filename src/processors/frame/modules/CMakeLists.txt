
add_subdirectory(face_swapper)
target_sources(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/face_enhancer.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_enhancer.h
)