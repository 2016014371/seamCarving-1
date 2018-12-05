cmake_minimum_required(VERSION 3.12)
project(seamCarving)


find_package(OpenCV)
set(CMAKE_CXX_STANDARD 14)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(seamCarving main.cpp)

target_link_libraries( seamCarving ${OpenCV_LIBS} )
