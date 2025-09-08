//
// Created by Yasen on 9/7/25.
//

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <iostream>
#include "matrices.h"

class ConvLayer {
  private:
    std::vector<Img> kernels;
    const uint8_t kernel_sz, stride, padding;
   public:
    ConvLayer(const uint8_t out_channels, const uint8_t kernel_sz, const uint8_t stride, const uint8_t padding);
    void info() const;
    Img activation(const std::vector<Img>& input) const;
};



#endif //CONV_LAYER_H
