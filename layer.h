//
// Created by Yasen on 9/7/25.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <Eigen/Dense>

class Layer {
public:
  virtual ~Layer() = default;

  virtual void Forward(const Eigen::VectorXf& input) = 0;

  virtual void Backward(const Eigen::VectorXf& prevActivation,
                        const Eigen::VectorXf& nextDerivative,
                        float learningRate) = 0;

  virtual const Eigen::VectorXf& Activation() = 0;

  virtual const Eigen::VectorXf& PreviousDerivative() = 0;

  virtual void SetWeights(Eigen::MatrixXf& new_weights) = 0;

  virtual void PrintInfo() const = 0;
};

#endif //LAYER_H
