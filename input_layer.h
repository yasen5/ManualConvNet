//
// Created by Yasen on 10/2/25.
//

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "layer.h"


class InputLayer final : public Layer {
public:
  void Forward(const Eigen::MatrixXd& input) override;

  void Backward(const Eigen::MatrixXd& prevActivation,
                const Eigen::MatrixXd& nextDerivative,
                double learningRate) override;

  const Eigen::MatrixXd& Activation() override;

  const Eigen::MatrixXd& PreviousDerrivative() override;

  void PrintInfo() const override;

  void SetWeights(Eigen::MatrixXd& new_weights) override;

  void SetInputs(const Eigen::MatrixXd& inputs);

private:
  const Eigen::MatrixXd* inputs_ = nullptr;
};


#endif //INPUT_LAYER_H
