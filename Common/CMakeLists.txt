add_library(Common dataset.cpp softmax_layer.cpp)
target_link_libraries(Common PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)