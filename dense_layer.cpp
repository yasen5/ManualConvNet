//
// Created by Yasen on 9/18/25.
//

#include "dense_layer.h"

#include <iostream>

void ReLU(Eigen::MatrixXd& input) {
  input = input.array().max(0);
}

DenseLayer::DenseLayer(int input_size, int output_size): weights_(
      Eigen::MatrixXd::Constant(output_size, input_size, 1.0)),
  biases_(Eigen::MatrixXd::Constant(output_size, 1, 0)),
  previous_derivative_(input_size, 1), activation_(output_size, 1) {
}

void DenseLayer::Forward(const Eigen::MatrixXd& input) {
  activation_ = weights_ * input + biases_;
  ReLU(activation_);
}

void DenseLayer::Backward(const Eigen::MatrixXd& prevActivation,
                          const Eigen::MatrixXd& nextDerivative,
                          const double learningRate) {
  weights_ += nextDerivative * prevActivation.transpose() * learningRate;
  biases_ += nextDerivative * learningRate;
  previous_derivative_ = weights_.transpose() * nextDerivative;
}

void DenseLayer::PrintInfo() const {
  std::cout << "Weight dims: " << weights_.rows() << " x " << weights_.cols() <<
      std::endl;
  std::cout << "Weights:\n" << weights_ << std::endl;
}