//
// Created by Yasen on 9/18/25.
//

#include "dense_layer.h"

#include <iostream>

#include "constants.h"

void ReLU(Eigen::VectorXf& input) {
  input = input.cwiseMax(0);
}

DenseLayer::DenseLayer(int input_size, int output_size): weights_(
      Eigen::MatrixXf::Random(output_size, input_size)),
  biases_(Eigen::VectorXf::Random(output_size)),
  previous_derivative_(input_size, 1), activation_(output_size, 1) {
  weights_ *= sqrt(2.0 / MLConstants::LinearConstants::INPUT_SIZE);
}

void DenseLayer::Forward(const Eigen::VectorXf& input) {
  activation_ = weights_ * input + biases_;
  ReLU(activation_);
}

void DenseLayer::Backward(const Eigen::VectorXf& prevActivation,
                          const Eigen::VectorXf& nextDerivative,
                          const float learningRate) {
  weights_ += nextDerivative * prevActivation.transpose() * learningRate;
  biases_ += nextDerivative * learningRate;
  // Eigen::ArrayXf arr1 = (activation_.array() > 0).cast<float>();
  // Eigen::ArrayXf arr2 = (weights_.transpose() * nextDerivative).array();
  // std::cout << "Arr1dims: " << arr1.rows() << " x " << arr1.cols() <<
  //     " arr2dims: " << arr2.rows() << " x " << arr2.cols() << std::endl;
  // Eigen::ArrayXf result = arr1 * arr2;
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
