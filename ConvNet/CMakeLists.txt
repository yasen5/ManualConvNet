add_library(ConvNet conv_layer.cpp matrices.cpp visualizer.cpp
        conv_net.cpp
        conv_net.h
        nd_layer.h
        conv_input.cpp
        conv_input.h
)
target_link_libraries(ConvNet PRIVATE Eigen3::Eigen Common ${OpenCV_Libs})
target_include_directories(ConvNet PRIVATE ${OpenCV_INCLUDE_DIRS})