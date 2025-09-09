//
// Created by Yasen on 9/8/25.
//

#ifndef MAXPOOL_LAYER_H
#define MAXPOOL_LAYER_H
#include <cstdint>

#include "layer.h"


class MaxpoolLayer : Layer {
private:
    const uint8_t padding, stride, kernel_sz;
public:
    MaxpoolLayer(uint8_t padding, uint8_t stride, uint8_t kernel_sz);
    [[nodiscard]] Img activation(const Img& input) const override;
};



#endif //MAXPOOL_LAYER_H
