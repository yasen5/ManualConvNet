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

  void Forward(const Eigen::MatrixXd& input) override;

  void Backward(const Eigen::MatrixXd& prevActivation,
                const Eigen::MatrixXd& nextDerivative,
                double learningRate) override;

  const Eigen::MatrixXd& Activation() override { return activation_; }

  const Eigen::MatrixXd& PreviousDerrivative() override {
    return previous_derivative_;
  }

  void PrintInfo() const override;

  void SetWeights(Eigen::MatrixXd& new_weights) override;

private:
  Eigen::MatrixXd weights_;
  Eigen::MatrixXd biases_;
  Eigen::MatrixXd previous_derivative_;
  Eigen::MatrixXd activation_;
};


#endif //DENSE_LAYER_H
