cmake_minimum_required(VERSION 3.10)

project(FaceDetectionProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(face_detect main.cpp)

target_link_libraries(face_detect ${OpenCV_LIBS})

set(MODEL_FILES
    "face_estimator.onnx"
)

foreach(FILENAME ${MODEL_FILES})
    configure_file(${FILENAME} ${CMAKE_CURRENT_BINARY_DIR}/${FILENAME} COPYONLY)
endforeach()
