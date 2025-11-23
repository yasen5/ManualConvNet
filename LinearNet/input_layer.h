//
// Created by Yasen on 10/2/25.
//

#ifndef INPUT_LAYER_H
#define INPUT_LAYER_H
#include "linear_layer.h"


class InputLayer final : public LinearLayer {
public:
  void Forward(const Eigen::VectorXf& input) override;

  void Backward(const Eigen::VectorXf& prevActivation,
                const Eigen::VectorXf& nextDerivative,
                float learningRate) override;

  const Eigen::VectorXf& Activation() override {
    return *inputs_;
  }

  const Eigen::VectorXf& PreviousDerivative() override;

  void PrintInfo() const override;

  void SetWeights(Eigen::MatrixXf& new_weights) override;

  void SetInputs(const Eigen::VectorXf& inputs);

private:
  const Eigen::VectorXf* inputs_ = nullptr;
};


#endif //INPUT_LAYER_H
