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
    static MatrixXf crossCorrelation(const Img& image, const Img& kernels, const uint8_t stride, const uint8_t padding);
    static float convolve(const MatrixXf& input, const MatrixXf& kernel);
    static MatrixXf maxPool(const MatrixXf& mat, const uint8_t padding, const uint8_t stride, const uint8_t kernel_sz);
    static void printImg(const Img& matrix);
    static Img initKernel(const uint8_t kernel_sz);
};

#endif //MATRICES_H
