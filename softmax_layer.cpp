//
// Created by Yasen on 10/3/25.
//

#include "softmax_layer.h"

#include <iostream>

SoftmaxLayer::SoftmaxLayer(const int num_inputs) : previous_derivative_(
                                                       num_inputs, 1),
                                                   activation_(num_inputs, 1) {
}


void SoftmaxLayer::Forward(const Eigen::MatrixXd& input) {
  double expSum = 0;
  for (int i = 0; i < input.rows(); i++) {
    expSum += exp(input(i));
  }
  for (int i = 0; i < input.rows(); i++) {
    activation_(i, 0) = exp(input(i)) / expSum;
  }
}

const Eigen::MatrixXd& SoftmaxLayer::Activation() {
  return activation_;
}

const Eigen::MatrixXd& SoftmaxLayer::PreviousDerrivative() {
  return previous_derivative_;
}

void SoftmaxLayer::PrintInfo() const {
  std::cout << "Softmax layer that takes in " << activation_.rows() << " inputs"
      << std::endl;
}

void SoftmaxLayer::SetWeights(Eigen::MatrixXd& new_weights) {
  std::cerr << "Setting weights for a softmax layer!" << std::endl;
}

void SoftmaxLayer::Backward(const Eigen::MatrixXd& prevActivation,
                            const Eigen::MatrixXd& nextDerivative,
                            double learningRate) {
  for (int i = 0; i < previous_derivative_.size(); i++) {
    for (int j = 0; j < previous_derivative_.size(); j++) {
      previous_derivative_(i, 0) += (i == j)
                                      ? nextDerivative(i, 0) * (1
                                          - nextDerivative(i, 0))
                                      : -nextDerivative(j, 0);
    }
  }
}





