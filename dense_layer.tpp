//
// Created by Yasen on 9/18/25.
//
#pragma once

#include "dense_layer.h"
#include <iostream>

inline void ReLU(Eigen::MatrixXd& input) {
  input = input.array().max(0);
}

template <int InputSize, int OutputSize>
void DenseLayer<InputSize, OutputSize>::Forward(
    const Eigen::Vector<float, InputSize>& input) {
  activation_ = weights_ * input + biases_;
  ReLU(activation_);
}

template <int InputSize, int OutputSize>
void DenseLayer<InputSize, OutputSize>::Backward(const double learningRate) {
  weights_ += *this->next_derivative_ * *this->previous_activation_.transpose()
      * learningRate;
  biases_ += *this->next_derivative_ * learningRate;
  previous_derivative_ = weights_.transpose() * *this->next_derivative_;
}

template <int InputSize, int OutputSize>
void DenseLayer<InputSize, OutputSize>::PrintInfo() const {
  std::cout << "Weight dims: " << weights_.rows() << " x " << weights_.cols() <<
      std::endl;
  std::cout << "Weights:\n" << weights_ << std::endl;
  std::cout << "Biases:\n" << biases_.transpose() << std::endl;
}

template <int InputSize, int OutputSize>
void DenseLayer<InputSize,
                OutputSize>::SetWeights(
    Eigen::Matrix<float, OutputSize, InputSize>& new_weights) {
  weights_ = std::move(new_weights);
}
