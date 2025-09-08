//
// Created by Yasen on 9/7/25.
//

#include "matrices.h"
#include <iostream>

MatrixXf Matrices::crossCorrelation(const Img& image, const Img& kernels, const uint8_t stride, const uint8_t padding) {
    const uint16_t kernel_sz = kernels.at(0).rows(); // Could be cols instead, should be n x n though
    const uint16_t convolvedRows = (image.at(0).rows() + 2 * padding - kernel_sz) / stride + 1;
    const uint16_t convolvedCols = (image.at(0).cols() + 2 * padding - kernel_sz) / stride + 1;
    MatrixXf convolved = MatrixXf::Zero(convolvedRows + 2 * padding, convolvedCols + 2 * padding);
    for (int i = 0; i < kernels.size(); i++) {
        MatrixXf oneColorImg = image.at(i);
        MatrixXf paddedImg = MatrixXf::Zero(oneColorImg.rows() + 2 * padding, oneColorImg.cols() + 2 * padding);
        paddedImg.block(padding, padding, oneColorImg.rows(), oneColorImg.cols()) = oneColorImg;
        const MatrixXf& kernel = kernels.at(i);
        for (int row = 0; row < convolvedRows; row++) {
            for (int col = 0; col < convolvedCols; col++) {
                convolved(row, col) += convolve(oneColorImg.block(row * stride, col * stride, kernel_sz, kernel_sz), kernel);
            }
        }
    }
    return convolved;
}

float Matrices::convolve(const MatrixXf& input, const MatrixXf& kernel) {
    if (kernel.rows() != input.rows() || kernel.cols() != input.cols()) {
        throw std::invalid_argument("Input size of (" + std::to_string(input.rows()) + ", " + std::to_string(input.cols()) + ") does not match input kernel size of (" + std::to_string(kernel.rows()) + ", " + std::to_string(kernel.cols()) + ")");
    }

    float sum = 0;
    for (int row = 0; row < kernel.rows(); row++) {
        for (int col = 0; col < kernel.cols(); col++) {
            sum += input(row, col) * kernel(row, col);
        }
    }
    return sum;
}

MatrixXf Matrices::maxPool(const MatrixXf& mat, const uint8_t padding, const uint8_t stride, const uint8_t kernel_sz) {
    const uint16_t convolvedRows = (mat.rows() + 2 * padding - kernel_sz) / stride + 1;
    const uint16_t convolvedCols = (mat.cols() + 2 * padding - kernel_sz) / stride + 1;
    MatrixXf pooled = MatrixXf::Zero(convolvedRows + 2 * padding, convolvedCols + 2 * padding);
    for (int row = 0; row < convolvedRows; row++) {
        for (int col = 0; col < convolvedCols; col++) {
            float greatest = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < kernel_sz; i++) {
                for (int j = 0; j < kernel_sz; j++) {
                    const float val = mat(row * stride + i, col * stride + j);
                    if (val > greatest) {
                        greatest = val;
                    }
                }
            }
            pooled(row, col) = greatest;
        }
    }
    return pooled;
}

void Matrices::printImg(const Img& matrix) {
    for (const MatrixXf& channel : matrix) {
        std::cout << channel << "\n" << std::endl;
    }
    std::cout << std::endl;
}

Img Matrices::initKernel(const uint8_t kernel_sz) {
    std::vector<MatrixXf> img(kernel_sz, MatrixXf::Random(kernel_sz, kernel_sz));
    return img;
}