target_sources(${ProgramName} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/face_analyser.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_analyser.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_detector_yolo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_detector_yolo.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_landmarker_68.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_landmarker_68.h
        ${CMAKE_CURRENT_SOURCE_DIR}/analyser_base.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/analyser_base.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_analyser_session.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_analyser_session.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_landmarker_68_5.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_landmarker_68_5.h
        ${CMAKE_CURRENT_SOURCE_DIR}/face_recognizer_arc.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/face_recognizer_arc.h
)
