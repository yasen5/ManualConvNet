//
// Created by Yasen on 10/2/25.
//

#include "input_layer.h"

#include <iostream>

void InputLayer::Forward(const Eigen::MatrixXd& input) {
  std::cerr << "Called Forward on an input layer!" << std::endl;
}

void InputLayer::Backward(const Eigen::MatrixXd& prevActivation,
                          const Eigen::MatrixXd& nextDerivative,
                          double learningRate) {
  std::cerr << "Called Backward on an input layer!" << std::endl;
}

const Eigen::MatrixXd& InputLayer::Activation() {
  return *inputs_;
}

const Eigen::MatrixXd& InputLayer::PreviousDerrivative() {
  std::cerr << "Asking for previous derivative from an input layer!" <<
      std::endl;
  std::exit(1);
}

void InputLayer::PrintInfo() const {
  std::cout << "Inputs transposed: " << inputs_->transpose() << std::endl;
}

void InputLayer::SetWeights(Eigen::MatrixXd& new_weights) {
  std::cerr << "Setting weights for an input layer!" << std::endl;
}

void InputLayer::SetInputs(const Eigen::MatrixXd& inputs) {
  inputs_ = &inputs;
}



