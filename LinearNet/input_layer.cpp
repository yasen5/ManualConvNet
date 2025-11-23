//
// Created by Yasen on 10/2/25.
//

#include "input_layer.h"

#include <iostream>

void InputLayer::Forward(const Eigen::VectorXf& input) {
  std::cerr << "Called Forward on an input layer!" << std::endl;
}

void InputLayer::Backward(const Eigen::VectorXf& prevActivation,
                          const Eigen::VectorXf& nextDerivative,
                          float learningRate) {
  std::cerr << "Called Backward on an input layer!" << std::endl;
}

const Eigen::VectorXf& InputLayer::PreviousDerivative() {
  std::cerr << "Asking for previous derivative from an input layer!" <<
      std::endl;
  std::exit(1);
}

void InputLayer::PrintInfo() const {
  std::cout << "Inputs: " << inputs_ << std::endl;
}

void InputLayer::SetWeights(Eigen::MatrixXf& new_weights) {
  std::cerr << "Setting weights for an input layer!" << std::endl;
}

void InputLayer::SetInputs(const Eigen::VectorXf& inputs) {
  inputs_ = &inputs;
}
