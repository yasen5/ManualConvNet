//
// Created by Yasen on 9/18/25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "layer.h"

template <int InputSize, int OutputSize>
class DenseLayer final : public Layer<InputSize, OutputSize> {
public:
  void Forward(const Eigen::Vector<float, InputSize>& input) override;

  void Backward(double learningRate) override;

  void SetWeights(
      Eigen::Matrix<float, OutputSize, InputSize>& new_weights) override;

  const Eigen::Vector<float, OutputSize> Activation() override {
    return activation_;
  }

  const Eigen::Vector<float, InputSize>& PreviousDerivative() override {
    return previous_derivative_;
  }

  void PrintInfo() const override;

  void SetWeights(Eigen::MatrixXd& new_weights) override;

private:
  Eigen::Matrix<float, OutputSize, InputSize> weights_;
  Eigen::Vector<float, OutputSize> biases_;
  Eigen::Vector<float, InputSize> previous_derivative_;
  Eigen::Vector<float, OutputSize> activation_;
};


#endif //DENSE_LAYER_H
