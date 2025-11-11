cmake_minimum_required(VERSION 3.29)
project(ManualConvNet)

if (MSVC)
    add_compile_options(/GR)      # Enable RTTI for MSVC
else ()
    add_compile_options(-frtti)   # Enable RTTI for GCC/Clang
endif ()

set(CMAKE_CXX_STANDARD 23)

find_package(Eigen3 5.0 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
# find_package(Matplot++ REQUIRED)

add_executable(ManualConvNet main.cpp)
add_subdirectory(LinearNet)
add_subdirectory(Common)

target_link_libraries(ManualConvNet
        PRIVATE
        Eigen3::Eigen
        ${OpenCV_LIBS}
        LinearNet
        Common
        #        Matplot++::matplot
)

target_include_directories(ManualConvNet
        PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tqdm.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sciplot/
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/
)
