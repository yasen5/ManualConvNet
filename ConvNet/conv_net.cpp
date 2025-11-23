//
// Created by Yasen on 11/22/25.
//

#include "conv_net.h"

#include "conv_input.h"

const Eigen::VectorXf& ConvNet::Predict() {
  conv_layers_.push_back(std::unique_ptr<ConvInput>());
}


void ConvNet::SetInputs(const Img& inputs) {
}


float ConvNet::Backprop(const Eigen::VectorXf& expected, float learning_rate) {
}
