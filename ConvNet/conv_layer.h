//
// Created by Yasen on 9/7/25.
//

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <iostream>

#include "matrices.h"
#include "nd_layer.h"

class ConvLayer final : public NDLayer {
public:
    ConvLayer(int in_channels, int out_channels, int kernel_sz, int stride, int padding);

    void Forward(const Img &input) override;

    void Backward(const Img &prevActivation,
                  const Img &nextDerivative,
                  float learningRate) override;

    const Img &Activation() override;

    const Img &PreviousDerivative() override;

    void SetWeights(std::vector<Img> &new_weights) override;

    void PrintInfo() const override;

private:
    std::vector<Img> kernels_;
    Eigen::VectorXf biases_;
    Img activation_;
    Img prev_derivative_;
    const int kernel_sz_, stride_, padding_;
};


#endif //CONV_LAYER_H
