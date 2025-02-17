cmake_minimum_required(VERSION 3.10)
project(YOLO_ana)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

add_executable(YOLO_ana main.cpp)
target_link_libraries(YOLO_ana ${OpenCV_LIBS} ${CUDA_LIBRARIES})
