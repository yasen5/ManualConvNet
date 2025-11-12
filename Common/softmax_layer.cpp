//
// Created by Yasen on 10/3/25.
//

#include "softmax_layer.h"

#include <iostream>

SoftmaxLayer::SoftmaxLayer(const int num_inputs) : previous_derivative_(
                                                       num_inputs),
                                                   activation_(num_inputs) {
}


void SoftmaxLayer::Forward(const Eigen::VectorXf& input) {
  activation_.setZero();
  Eigen::VectorXf shifted = input;
  shifted = shifted.array() - shifted.maxCoeff();
  float expSum = 0;
  for (int i = 0; i < input.rows(); i++) {
    expSum += exp(shifted(i));
  }
  for (int i = 0; i < input.rows(); i++) {
    activation_(i, 0) = exp(shifted(i)) / expSum;
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
  previous_derivative_ = -1 * nextDerivative;
  // TODO find out why this needs to be negative
  // THE ABOVE CODE ASSUMES THAT THE NET IS APPLYING CROSS-ENTROPY LOSS
  // If that loss function is not used, and the resulting derivative isn't
  // as simplified as `prediction - expected` or something like that, uncomment
  // the code below
  /*for (int i = 0; i < previous_derivative_.size(); i++) {
    for (int j = 0; j < previous_derivative_.size(); j++) {
      const float calculated = (i == j)
                                 ? nextDerivative(i) * (
                                     1 - nextDerivative(i))
                                 : nextDerivative(i) * -nextDerivative(j);
      previous_derivative_(i) += calculated;
    }
  }*/
}





