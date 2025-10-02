#include <iostream>
// #include "dataset.h"
// #include <opencv2/core/eigen.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/opencv.hpp>

// #include "conv_layer.h"
// #include "conv_net.h"
#include "dense_layer.h"
#include "dense_net.h"
// #include "matrices.h"
// #include "maxpool_layer.h"
// #include "visualizer.h"

// using Eigen::MatrixXf;

int main() {
    DenseNet net;
    net.AddLayer(std::make_unique<DenseLayer>(2, 2));
    net.AddLayer(std::make_unique<DenseLayer>(2, 2));
    net.AddLayer(std::make_unique<DenseLayer>(2, 1));
    net.PrintInfo();
    Eigen::MatrixXd input(2, 1);
    input << 1, 1;
    std::cout << "prediction: " << net.Predict(input) << std::endl;
    Eigen::MatrixXd expected(1, 1);
    expected << 4;
    net.Backprop(expected, input, 0.01);
    net.PrintInfo();
    // std::vector<std::unique_ptr<Layer>> layers;
    // layers.push_back(std::make_unique<ConvLayer>(3, 2, 3, 1, 0));
    // ConvNet net(std::move(layers));
    // Dataset data("/Users/yasen/ClionProjects/ManualConvNet/Data/train.csv", "/Users/yasen/ClionProjects/ManualConvNet/Data/literally nothing lol", "/Users/yasen/ClionProjects/ManualConvNet/Data/test.csv");
    // net.predict(std::vector<MatrixXf>{data.getData(TRAIN).at(0).img});
    exit(0);
}