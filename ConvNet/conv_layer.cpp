//
// Created by Yasen on 9/7/25.
//

#include "conv_layer.h"

ConvLayer::ConvLayer(const int in_channels, const int out_channels,
                     const int kernel_sz, const int stride,
                     const int padding) : kernels_(out_channels),
                                          biases_(
                                              Eigen::VectorXf::Constant(
                                                  out_channels, 1000)),
                                          activation_(out_channels),
                                          prev_derivative_(in_channels),
                                          kernel_sz_(kernel_sz),
                                          stride_(stride),
                                          padding_(padding) {
  for (int i = 0; i < out_channels; i++) {
    kernels_[i] = Matrices::initKernel(in_channels, kernel_sz);
  }
}

void ConvLayer::PrintInfo() const {
  std::cout << "Kernels: " << kernels_.size() << "\tStride: " <<
      stride_ << "\tPadding: " << padding_ << std::endl;
  int kernelCounter = 0;
  for (const Img& img : kernels_) {
    kernelCounter++;
    std::cout << "Kernel " << kernelCounter << " : " << std::endl;
    Matrices::printImg(img);
  }
}

void ConvLayer::Forward(const Img& input) {
  for (size_t i = 0; i < input.size(); i++) {
    activation_[i] = Matrices::crossCorrelation(input, kernels_[i], stride_,
                                                padding_, false);
    activation_[i].array() += biases_(i);
  }
}

const Img& ConvLayer::Activation() {
  return activation_;
}

const Img& ConvLayer::PreviousDerivative() {
  return prev_derivative_;
}

void ConvLayer::SetWeights(std::vector<Img>& new_weights) {
  kernels_ = new_weights;
}

void ConvLayer::Backward(const Img& prevActivation, const Img& nextDerivative,
                         float learningRate) {
}

