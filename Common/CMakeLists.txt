#add_library(Common dataset.cpp softmax_layer.cpp)
add_library(Common dataset.cpp softmax_layer.cpp)
target_link_libraries(Common PRIVATE Eigen3::Eigen)