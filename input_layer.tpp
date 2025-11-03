//
// Created by Yasen on 10/2/25.
//
#pragma once

#include "input_layer.h"

#include <iostream>

template <int InputSize, int OutputSize>
void InputLayer<InputSize, OutputSize>::Forward(
    const Eigen::Vector<float, InputSize>& input) {
  std::cerr << "Called Forward on an input layer!" << std::endl;
}

template <int InputSize, int OutputSize>
void InputLayer<InputSize, OutputSize>::Backward(double learningRate) {
  std::cerr << "Called Backward on an input layer!" << std::endl;
}

template <int InputSize, int OutputSize>
const Eigen::Vector<float, OutputSize>& InputLayer<
  InputSize, OutputSize>::Activation() {
  return *inputs_;
}

template <int InputSize, int OutputSize>
const Eigen::Vector<float, InputSize>& InputLayer<
  InputSize, OutputSize>::PreviousDerivative() {
  std::cerr << "Asking for previous derivative from an input layer!" <<
      std::endl;
  std::exit(1);
}

template <int InputSize, int OutputSize>
void InputLayer<InputSize, OutputSize>::PrintInfo() const {
  std::cout << "Inputs transposed: " << inputs_->transpose() << std::endl;
}

template <int InputSize, int OutputSize>
void InputLayer<InputSize,
                OutputSize>::SetWeights(
    Eigen::Matrix<float, OutputSize, InputSize>& new_weights) {
  std::cerr << "Setting weights for an input layer!" << std::endl;
}

template <int InputSize, int OutputSize>
void InputLayer<InputSize,
                OutputSize>::SetInputs(const Eigen::MatrixXd& inputs) {
  inputs_ = &inputs;
}



