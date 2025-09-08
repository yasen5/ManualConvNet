//
// Created by Yasen on 9/7/25.
//

#include "conv_layer.h"

ConvLayer::ConvLayer(const uint16_t in_channels, const uint16_t out_channels, const uint8_t kernel_sz, const uint8_t stride, const uint8_t padding) : kernels(out_channels), kernel_sz(kernel_sz), stride(stride), padding(padding) {
    for (int i = 0; i < out_channels; i++) {
        kernels[i] = Matrices::initKernel(in_channels, kernel_sz);
    }
}

void ConvLayer::info() const {
    std::cout << "Kernels: " << kernels.size() << "\tStride: " << static_cast<int>(stride) << "\tPadding: " << static_cast<int>(padding) << std::endl;
    int kernelCounter = 0;
    for (const Img& img : kernels) {
        kernelCounter++;
        std::cout << "Kernel " << kernelCounter << " : " << std::endl;
        Matrices::printImg(img);
    }
}

Img ConvLayer::activation(const std::vector<Img>& input) const {
    Img output;
    output.reserve(kernels.size());
    for (const Img& kernel : kernels) {
        for (const Img& img : input) {
            output.push_back(Matrices::crossCorrelation(img, kernel, stride, padding));
        }
    }
    return output;
}