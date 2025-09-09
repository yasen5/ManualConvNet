//
// Created by Yasen on 9/8/25.
//

#include "maxpool_layer.h"

MaxpoolLayer::MaxpoolLayer(const uint8_t padding, const uint8_t stride, const uint8_t kernel_sz) : padding(padding), stride(stride), kernel_sz(kernel_sz) {}

Img MaxpoolLayer::activation(const Img& input) const {
    const uint16_t convolvedRows = (input.at(0).rows() + 2 * padding - kernel_sz) / stride + 1;
    const uint16_t convolvedCols = (input.at(0).cols() + 2 * padding - kernel_sz) / stride + 1;
    Img pooled = Img(input.size(), MatrixXf::Zero(convolvedRows, convolvedCols));
    for (int i = 0; i < input.size(); i++) {
        const MatrixXf& channel = input.at(i);
        for (int row = 0; row < channel.rows() - kernel_sz + 1; row++) {
            for (int col = 0; col < channel.cols() - kernel_sz + 1; col++) {
                pooled.at(i)(row, col) = channel.block(row, col, kernel_sz, kernel_sz).maxCoeff();
            }
        }
    }
    return pooled;
}
