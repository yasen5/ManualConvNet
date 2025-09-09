//
// Created by Yasen on 9/7/25.
//

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <iostream>

#include "layer.h"
#include "matrices.h"

class ConvLayer : Layer {
  private:
    std::vector<Img> kernels;
    const uint8_t kernel_sz, stride, padding;
   public:
    ConvLayer(uint16_t in_channels, uint16_t out_channels, uint8_t kernel_sz, uint8_t stride, uint8_t padding);
    void info() const;
    [[nodiscard]] Img activation(const Img& input) const override;
};



#endif //CONV_LAYER_H
