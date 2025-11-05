//
// Created by Yasen on 9/18/25.
//

#include "dense_layer.h"

#include <iostream>

void ReLU(Eigen::VectorXf& input) {
  input = input.array().max(0);
}

DenseLayer::DenseLayer(int input_size, int output_size): weights_(
      Eigen::MatrixXf::Random(output_size, input_size)),
  //biases_(Eigen::MatrixXf::Constant(output_size, 1, 0))
  biases_(Eigen::MatrixXf::Random(output_size, 1)),
  previous_derivative_(input_size, 1), activation_(output_size, 1) {
}

void DenseLayer::Forward(const Eigen::VectorXf& input) {
  activation_.setZero();
  activation_ = weights_ * input + biases_;
  ReLU(activation_);
}

void DenseLayer::Backward(const Eigen::VectorXf& prevActivation,
                          const Eigen::VectorXf& nextDerivative,
                          const float learningRate) {
  weights_ += nextDerivative * prevActivation.transpose() * learningRate;
  biases_ += nextDerivative * learningRate;
  previous_derivative_ = weights_.transpose() * nextDerivative;
}

void DenseLayer::PrintInfo() const {
  std::cout << "Weight dims: " << weights_.rows() << " x " << weights_.cols() <<
      std::endl;
  std::cout << "Weights:\n" << weights_ << std::endl;
  std::cout << "Biases:\n" << biases_.transpose() << std::endl;
}

void DenseLayer::SetWeights(Eigen::MatrixXf& new_weights) {
  weights_ = std::move(new_weights);
}
