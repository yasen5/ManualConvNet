//
// Created by Yasen on 10/2/25.
//

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "layer.h"

template <int InputSize, int OutputSize>
class InputLayer final : public Layer<InputSize, OutputSize> {
public:
  void Forward(const Eigen::Vector<float, InputSize>& input) override;

  void Backward(double learningRate) override;

  const Eigen::Vector<float, OutputSize>& Activation() override;

  const Eigen::Vector<float, InputSize>& PreviousDerivative() override;

  void SetWeights(
      Eigen::Matrix<float, OutputSize, InputSize>& new_weights) override;

  void SetInputs(const Eigen::MatrixXd& inputs);

  void PrintInfo() const override;

private:
  const Eigen::MatrixXd* inputs_ = nullptr;
};


#endif //INPUT_LAYER_H
