//
// Created by Yasen on 9/7/25.
//

#include "conv_layer.h"

ConvLayer::ConvLayer(const int in_channels, const int out_channels,
                     const int kernel_sz, const int stride,
                     const int padding) : kernels_(out_channels),
                                          biases_(
                                              Eigen::VectorXf::Random(
                                                  out_channels)),
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
  }
}

void ConvLayer::Forward(const Img& input) {
  for (size_t out_channels = 0; out_channels < kernels_.size(); out_channels
       ++) {
    activation_[out_channels] = Matrices::CrossCorrelation(
        input, kernels_[out_channels], stride_,
        padding_, false);
    activation_[out_channels].array() += biases_(out_channels);
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

Img ConvLayer::ImgBlock(const Img& img, const int startRow, const int startCol,
                        const int rows, const int cols) {
  Img block(img.size());
  for (size_t i = 0; i < img.size(); i++) {
    block[i] = img[i].block(startRow, startCol, rows, cols);
  }
  return block;
}

void ConvLayer::ScaleImg(Img& img, const float scalar) {
  for (int i = 0; i < img.size(); i++) {
    img[i] *= scalar;
  }
}

void ConvLayer::AddImages(Img& operand, const Img& img2) {
  for (size_t i = 0; i < operand.size(); i++) {
    operand[i] += img2[i];
  }
}

void ConvLayer::Backward(const Img& prevActivation, const Img& nextDerivative,
                         float learningRate) {
  prev_derivative_ = Img(prevActivation.size(), Eigen::MatrixXf::Zero(
                             prevActivation[0].rows(),
                             prevActivation[0].cols()));
  for (int input_channel = 0; input_channel < prevActivation.size();
       input_channel++) {
    for (int output_channel = 0; output_channel < nextDerivative.size();
         output_channel++) {
      prev_derivative_[input_channel] += Matrices::FullConvolve(
          nextDerivative[output_channel],
          kernels_[output_channel][input_channel], stride_,
          padding_, false);
    }
  }
  for (size_t output_channel = 0; output_channel < nextDerivative.size();
       output_channel++) {
    const int row_steps = (prevActivation.at(0).rows() + 2 * padding_ -
                           kernel_sz_) / stride_ + 1;
    const int col_steps = (prevActivation.at(0).cols() + 2 * padding_ -
                           kernel_sz_) / stride_ + 1;
    for (int row = 0; row < row_steps; row++) {
      for (int col = 0; col < col_steps; col++) {
        Img kernel_derivative = ImgBlock(
            prevActivation, row, col,
            kernel_sz_, kernel_sz_);
        ScaleImg(kernel_derivative, nextDerivative[output_channel](
                     row, col));
        AddImages(kernels_[output_channel], kernel_derivative);
      }
    }
  }
}
