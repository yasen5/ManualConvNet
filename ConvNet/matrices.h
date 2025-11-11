//
// Created by Yasen on 9/7/25.
//

#ifndef MATRICES_H
#define MATRICES_H

#include <Eigen/Dense>

typedef std::vector<Eigen::MatrixXf> Img;
using Eigen::MatrixXf;

class Matrices {
public:
    static MatrixXf crossCorrelation(const Img& image, const Img& kernels, uint8_t stride, uint8_t padding);
    static float convolve(const MatrixXf& input, const MatrixXf& kernel);
    static MatrixXf maxPool(const MatrixXf& mat, uint8_t padding, uint8_t stride, uint8_t kernel_sz);
    static void printImg(const Img& matrix);
    static Img initKernel(const uint16_t input_channels, const uint8_t kernel_sz) {
        Img kernel(input_channels, MatrixXf::Random(kernel_sz, kernel_sz));
        return kernel;
    }
};

#endif //MATRICES_H
