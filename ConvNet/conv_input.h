//
// Created by Yasen on 11/22/25.
//

#ifndef CONV_INPUT_H
#define CONV_INPUT_H
#include "nd_layer.h"


class ConvInput final : public NDLayer {
public:
  void Forward(const Img& input) override;

  void Backward(const Img& prevActivation,
                const Img& nextDerivative,
                float learningRate) override;

  const Img& Activation() override;

  const Img& PreviousDerivative() override;

  void SetWeights(Img& new_weights) override;

  void PrintInfo() const override;

private:
  const Img* inputs_ = nullptr;
};


#endif //CONV_INPUT_H
