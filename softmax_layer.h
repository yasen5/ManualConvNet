//
// Created by Yasen on 10/3/25.
//

#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "layer.h"


class SoftmaxLayer final : public Layer {
public:
  SoftmaxLayer(int num_inputs);

  void Forward(const Eigen::MatrixXd& input) override;

  void Backward(const Eigen::MatrixXd& prevActivation,
                const Eigen::MatrixXd& nextDerivative,
                double learningRate) override;

  const Eigen::MatrixXd& Activation() override;

  const Eigen::MatrixXd& PreviousDerrivative() override;

  void SetWeights(Eigen::MatrixXd& new_weights) override;

  void PrintInfo() const override;

private:
  Eigen::MatrixXd previous_derivative_;
  Eigen::MatrixXd activation_;
};


#endif //SOFTMAX_LAYER_H
