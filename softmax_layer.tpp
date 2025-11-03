//
// Created by Yasen on 10/3/25.
//
#pragma once

#include "softmax_layer.h"

#include <iostream>

template <int InputSize, int OutputSize>
void SoftmaxLayer<InputSize, OutputSize>::Forward(
    const Eigen::Vector<float, InputSize>& input) {
  double expSum = 0;
  for (int i = 0; i < input.rows(); i++) {
    expSum += exp(input(i));
  }
  for (int i = 0; i < input.rows(); i++) {
    activation_(i, 0) = exp(input(i)) / expSum;
  }
}

template <int InputSize, int OutputSize>
const Eigen::Vector<float, OutputSize>& SoftmaxLayer<
  InputSize, OutputSize>::Activation() {
  return activation_;
}

template <int InputSize, int OutputSize>
const Eigen::Vector<float, InputSize>& SoftmaxLayer<
  InputSize, OutputSize>::PreviousDerivative() {
  return previous_derivative_;
}

template <int InputSize, int OutputSize>
void SoftmaxLayer<InputSize, OutputSize>::PrintInfo() const {
  std::cout << "Softmax layer that takes in " << activation_.rows() << " inputs"
      << std::endl;
}

template <int InputSize, int OutputSize>
void SoftmaxLayer<InputSize, OutputSize>::SetWeights(
    Eigen::Matrix<float, OutputSize, InputSize>& new_weights) {
  std::cerr << "Setting weights for a softmax layer!" << std::endl;
}

template <int InputSize, int OutputSize>
void SoftmaxLayer<InputSize, OutputSize>::Backward(double learningRate) {
  for (int i = 0; i < previous_derivative_.size(); i++) {
    for (int j = 0; j < previous_derivative_.size(); j++) {
      double calculated = (i == j)
                            ? *this->next_derivative(i) * (
                                1 - *this->next_derivative_(i))
                            : -*this->next_derivative_(j);
      previous_derivative_(i, 0) += calculated;
    }
  }
}





