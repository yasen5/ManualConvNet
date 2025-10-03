cmake_minimum_required(VERSION 3.29)
project(ManualConvNet)

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)

add_executable(ManualConvNet main.cpp dataset.cpp visualizer.cpp
#        visualizer.h
#        matrices.cpp
#        conv_layer.h
#        conv_layer.cpp
        layer.h
#        maxpool_layer.h
#        maxpool_layer.cpp
#        conv_net.h
#        conv_net.cpp
        dense_net.h
        dense_net.cpp
        dense_layer.h
        dense_layer.cpp
        input_layer.h
        input_layer.cpp
)

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
