//
// Created by Yasen on 10/3/25.
//

#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"

template <int InputSize, int OutputSize>
class SoftmaxLayer final : public Layer<InputSize, OutputSize> {
public:
  void Forward(const Eigen::Vector<float, InputSize>& input) override;

  void Backward(double learningRate) override;

  const Eigen::Vector<float, OutputSize>& Activation() override;

  const Eigen::Vector<float, InputSize>& PreviousDerivative() override;

  void SetWeights(
      Eigen::Matrix<float, OutputSize, InputSize>& new_weights) override;

  void PrintInfo() const override;

private:
  Eigen::Vector<float, InputSize> previous_derivative_;
  Eigen::Vector<float, OutputSize> activation_;
};


#endif //SOFTMAX_LAYER_H
