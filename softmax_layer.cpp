//
// Created by Yasen on 10/3/25.
//

#include "softmax_layer.h"

#include <iostream>

SoftmaxLayer::SoftmaxLayer(const int num_inputs) : previous_derivative_(
                                                       num_inputs, 1),
                                                   activation_(num_inputs, 1) {
}


void SoftmaxLayer::Forward(const Eigen::VectorXf& input) {
  float expSum = 0;
  for (int i = 0; i < input.rows(); i++) {
    expSum += exp(input(i));
  }
  for (int i = 0; i < input.rows(); i++) {
    activation_(i, 0) = exp(input(i)) / expSum;
  }
}

void SoftmaxLayer::PrintInfo() const {
  std::cout << "Softmax layer that takes in " << activation_.rows() << " inputs"
      << std::endl;
}

void SoftmaxLayer::SetWeights(Eigen::MatrixXf& new_weights) {
  std::cerr << "Setting weights for a softmax layer!" << std::endl;
}

void SoftmaxLayer::Backward(const Eigen::VectorXf& prevActivation,
                            const Eigen::VectorXf& nextDerivative,
                            float learningRate) {
  for (int i = 0; i < previous_derivative_.size(); i++) {
    for (int j = 0; j < previous_derivative_.size(); j++) {
      const float calculated = (i == j)
                                 ? nextDerivative(i, 0) * (
                                     1 - nextDerivative(i, 0))
                                 : -nextDerivative(j, 0);
      previous_derivative_(i, 0) += calculated;
    }
  }
}





