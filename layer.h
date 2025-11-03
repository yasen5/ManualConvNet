//
// Created by Yasen on 9/7/25.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <Eigen/Dense>

template <int InputSize, int OutputSize>
class Layer {
public:
  const Eigen::Vector<float, OutputSize>* next_derivative_ = nullptr;
  const Eigen::Vector<float, InputSize>* previous_activation_ = nullptr;

  virtual ~Layer() = default;

  virtual void Forward(const Eigen::Vector<float, InputSize>& input) = 0;

  virtual void Backward(double learningRate) = 0;

  virtual const Eigen::Vector<float, OutputSize>& Activation() = 0;

  virtual const Eigen::Vector<float, InputSize>& PreviousDerivative() = 0;

  virtual void SetWeights(
      Eigen::Matrix<float, OutputSize, InputSize>& new_weights) = 0;

  virtual void PrintInfo() const = 0;
};

#endif //LAYER_H
