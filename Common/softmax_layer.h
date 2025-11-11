//
// Created by Yasen on 10/3/25.
//

#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "../LinearNet/linear_layer.h"


class SoftmaxLayer final : public Layer {
public:
  SoftmaxLayer(int num_inputs);

  void Forward(const Eigen::VectorXf& input) override;

  void Backward(const Eigen::VectorXf& prevActivation,
                const Eigen::VectorXf& nextDerivative,
                float learningRate) override;

  const Eigen::VectorXf& Activation() override {
    return activation_;
  }

  const Eigen::VectorXf& PreviousDerivative() override {
    return previous_derivative_;
  }

  void SetWeights(Eigen::MatrixXf& new_weights) override;

  void PrintInfo() const override;

private:
  Eigen::VectorXf previous_derivative_;
  Eigen::VectorXf activation_;
};


#endif //SOFTMAX_LAYER_H
