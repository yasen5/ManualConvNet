//
// Created by Yasen on 9/7/25.
//

#ifndef MATRICES_H
#define MATRICES_H

#include <Eigen/Dense>
#include "../Common/constants.h"

typedef std::vector<Eigen::MatrixXf> Img;
using Eigen::MatrixXf;

class Matrices {
public:
  static MatrixXf CrossCorrelation(const Img& image, const Img& kernel,
                                   int stride, int padding,
                                   bool verbose);

  static float FrobeniusInner(const MatrixXf& input, const MatrixXf& kernel);

  static MatrixXf FullConvolve(const MatrixXf& mat, const MatrixXf& kernel,
                               int stride, int padding, bool verbose);

  static MatrixXf CrossCorrelate(const MatrixXf& mat, const MatrixXf& kernel,
                                 int stride, int padding, bool verbose);

  static MatrixXf maxPool(const MatrixXf& mat, uint8_t padding, uint8_t stride,
                          uint8_t kernel_sz);

  static void PrintImg(const Img& matrix);

  static void PrintDims(std::string img_name, const Img& img);

  static Img initKernel(const uint16_t input_channels,
                        const uint8_t kernel_sz) {
    Img kernel(input_channels,
               MatrixXf::Random(kernel_sz, kernel_sz) * sqrt(
                   2.0 / MLConstants::ConvConstants::INPUT_SIZE));
    return kernel;
  }

  static Eigen::VectorXf Flatten(const Img& convImg);

  static Img Unflatten(const Eigen::VectorXf& vec, int channels);
};

#endif //MATRICES_H
