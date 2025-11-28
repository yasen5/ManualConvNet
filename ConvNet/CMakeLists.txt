add_library(ConvNet conv_layer.cpp matrices.cpp visualizer.cpp
        conv_net.cpp
        conv_net.h
        nd_layer.h
)
target_link_libraries(ConvNet PRIVATE Eigen3::Eigen Common ${OpenCV_Libs})
target_include_directories(ConvNet PRIVATE ${OpenCV_INCLUDE_DIRS})