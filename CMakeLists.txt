cmake_minimum_required(VERSION 3.29)
project(ManualConvNet)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_executable(ManualConvNet main.cpp dataset.cpp visualizer.cpp
        visualizer.h)

# Modern way (Eigen uses imported target, OpenCV often does not)
target_link_libraries(ManualConvNet
        PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}   # this is a list of libraries to link
)

target_include_directories(ManualConvNet
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)
