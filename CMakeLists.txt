cmake_minimum_required(VERSION 3.29)
project(ManualConvNet)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

add_executable(ManualConvNet main.cpp)

target_link_libraries(ManualConvNet PRIVATE Eigen3::Eigen)
