//
// Created by Yasen on 9/18/25.
//

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H
#include <vector>

#include "layer.h"

class DenseLayer final : public Layer {
public:
  DenseLayer(int input_size, int output_size);

  void Forward(const Eigen::VectorXf& input) override;

  void Backward(const Eigen::VectorXf& prevActivation,
                const Eigen::VectorXf& nextDerivative,
                float learningRate) override;

  const Eigen::VectorXf& Activation() override { return activation_; }

  const Eigen::VectorXf& PreviousDerivative() override {
    return previous_derivative_;
  }

  void PrintInfo() const override;

  void SetWeights(Eigen::MatrixXf& new_weights) override;

private:
  Eigen::MatrixXf weights_;
  Eigen::VectorXf biases_;
  Eigen::VectorXf previous_derivative_;
  Eigen::VectorXf activation_;
};


#endif //DENSE_LAYER_H
