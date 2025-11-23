add_library(ConvNet conv_layer.cpp matrices.cpp maxpool_layer.cpp visualizer.cpp
        conv_net.cpp
        conv_net.h
        nd_layer.h
        conv_input.cpp
        conv_input.h
)
target_link_libraries(ConvNet Eigen3::Eigen Common)