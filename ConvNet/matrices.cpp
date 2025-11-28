//
// Created by Yasen on 9/7/25.
//

#include "matrices.h"
#include <iostream>

MatrixXf Matrices::CrossCorrelation(const Img& image, const Img& kernel,
                                    const int stride,
                                    const int padding, const bool verbose) {
  if (verbose) {
    PrintDims("Image", image);
    PrintDims("Kernel", kernel);
  }
  Eigen::MatrixXf mat = CrossCorrelate(image[0], kernel[0], stride, padding,
                                       verbose);
  for (int channel = 0; channel < kernel.size(); channel++) {
    mat += CrossCorrelate(image[channel], kernel[channel], stride, padding,
                          verbose);
  }
  return mat;
}

float Matrices::FrobeniusInner(const MatrixXf& input, const MatrixXf& kernel) {
  if (kernel.rows() != input.rows() || kernel.cols() != input.cols()) {
    throw std::invalid_argument(
        "Input size of (" + std::to_string(input.rows()) + ", " +
        std::to_string(input.cols()) + ") does not match input kernel size of ("
        + std::to_string(kernel.rows()) + ", " + std::to_string(kernel.cols()) +
        ")");
  }

  float sum = 0;
  for (int row = 0; row < kernel.rows(); row++) {
    for (int col = 0; col < kernel.cols(); col++) {
      sum += input(row, col) * kernel(row, col);
    }
  }
  return sum;
}

void Matrices::PrintImg(const Img& matrix) {
  for (const MatrixXf& channel : matrix) {
    std::cout << channel << "\n" << std::endl;
  }
  std::cout << std::endl;
}

void Matrices::PrintDims(std::string img_name, const Img& img) {
  std::cout << img_name << " dims: ";
  std::cout << "Channels: " << img.size() << "\tRows: " << img[0].rows() <<
      "\tCols: " << img[0].cols() << std::endl;
}

MatrixXf Matrices::FullConvolve(const MatrixXf& mat, const MatrixXf& kernel,
                                int stride, int padding, bool verbose) {
  const int side_buffer = kernel.rows() - 1;
  const Eigen::MatrixXf flippedKernel = kernel.colwise().reverse().rowwise().
      reverse();
  return CrossCorrelate(mat, flippedKernel, stride, padding + side_buffer,
                        verbose);
}

MatrixXf Matrices::CrossCorrelate(const MatrixXf& mat, const MatrixXf& kernel,
                                  int stride, int padding, bool verbose) {
  const int kernel_sz = kernel.rows();
  const int mat_sz = mat.rows();
  const int crossCorrelatedDims =
      (mat_sz + 2 * padding - kernel_sz) / stride + 1;
  MatrixXf correlated = MatrixXf::Zero(crossCorrelatedDims,
                                       crossCorrelatedDims);
  MatrixXf padded = MatrixXf::Zero(mat_sz + 2 * padding, mat_sz + 2 * padding);
  padded.block(padding, padding, mat_sz, mat_sz) = mat;
  for (int row = 0; row < crossCorrelatedDims; row++) {
    for (int col = 0; col < crossCorrelatedDims; col++) {
      correlated(row, col) = FrobeniusInner(
          padded.block(row * stride, col * stride, kernel_sz, kernel_sz),
          kernel);
    }
  }
  return correlated;
}

Eigen::VectorXf Matrices::Flatten(const Img& convImg) {
  const int flattened_size = convImg[0].cols() * convImg[0].
                             rows();
  Eigen::VectorXf flattened(flattened_size * convImg.size());
  int offset = 0;
  for (const Eigen::MatrixXf& channel : convImg) {
    flattened.segment(offset, channel.size()) = Eigen::Map<const
      Eigen::VectorXf>(
        channel.data(), channel.size());
    offset += channel.size();
  }
  return flattened;
}

Img Matrices::Unflatten(const Eigen::VectorXf& vec,
                        const int channels) {
  const int channel_size = vec.size() / channels;
  const int new_dim = sqrt(channel_size);
  Img unflattened(channels);
  for (int i = 0; i < channels; i++) {
    unflattened[i] = Eigen::Map<const
      Eigen::MatrixXf>(vec.data() + i * channel_size, new_dim, new_dim);
  }
  return unflattened;
}