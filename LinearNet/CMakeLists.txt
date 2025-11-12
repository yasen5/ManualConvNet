#add_library(LinearNet dense_layer.cpp dense_net.cpp input_layer.cpp)
add_library(LinearNet dense_layer.cpp dense_net.cpp input_layer.cpp)
target_link_libraries(LinearNet PRIVATE Eigen3::Eigen Common)