//
// Created by Yasen on 9/18/25.
//

#include "dense_layer.h"

#include <iostream>

#include "../Common/constants.h"

void ReLU(Eigen::VectorXf& input) {
  input = input.cwiseMax(0);
}

DenseLayer::DenseLayer(int input_size, int output_size) : weights_(
      Eigen::MatrixXf::Random(output_size, input_size)),
  biases_(Eigen::VectorXf::Random(output_size)),
  previous_derivative_(input_size), activation_(output_size) {
  weights_ *= sqrt(2.0 / MLConstants::LinearConstants::INPUT_SIZE);
}

void DenseLayer::Forward(const Eigen::VectorXf& input) {
  activation_ = weights_ * input + biases_;
  ReLU(activation_);
}

void DenseLayer::Backward(const Eigen::VectorXf& prevActivation,
                          const Eigen::VectorXf& nextDerivative,
                          const float learningRate) {
  weights_ += nextDerivative * prevActivation.transpose();
  biases_ += nextDerivative;
  previous_derivative_ = (weights_.transpose() * nextDerivative).array() *
                         (prevActivation.array() > 0).cast<float>();
}

void DenseLayer::PrintInfo() const {
  std::cout << "Weight dims: " << weights_.rows() << " x " << weights_.cols() <<
      std::endl;
  std::cout << "Weights:\n" << weights_ << std::endl;
  std::cout << "Biases:\n" << biases_.transpose() << std::endl;
}

void DenseLayer::SetWeights(Eigen::MatrixXf& new_weights) {
  weights_ = new_weights;
}
