//
// Created by Yasen on 9/17/25.
//

#include "dense_net.h"
#include <iostream>

#include "softmax_layer.h"

const Eigen::VectorXf& DenseNet::Predict(const Eigen::VectorXf& input) {
  layers_[0]->Forward(input);
  for (int i = 1; i < layers_.size(); i++) {
    layers_[i]->Forward(layers_[i - 1]->Activation());
    for (const float val : layers_[i]->Activation()) {
      if (std::isnan(val)) {
        std::cerr << "Generated nan value in output of layer: " << i <<
            std::endl;
        exit(0);
      }
    }
  }
  return layers_[layers_.size() - 1]->Activation();
}

float DenseNet::Backprop(const Eigen::VectorXf& input,
                         const Eigen::VectorXf& expected,
                         const float learning_rate) {
  const Eigen::VectorXf pred = Predict(input);
  const Eigen::VectorXf loss_derivative = (pred - expected) * learning_rate;
  layers_[layers_.size() - 1]->Backward(
      layers_[layers_.size() - 2]->Activation(),
      loss_derivative, learning_rate);
  for (int i = layers_.size() - 2; i >= 0; i--) {
    layers_[i]->Backward((i == 0) ? input : layers_.at(i - 1)->Activation(),
                         layers_.at(i + 1)->PreviousDerivative(),
                         learning_rate);
  }
  const float cross_entropy_loss = -1 * (expected.array() * log(pred.array())).
                                   sum();
  return cross_entropy_loss;
}

void DenseNet::AddLayer(std::unique_ptr<LinearLayer>&& layer) {
  layers_.push_back(std::move(layer));
}

void DenseNet::PrintInfo() const {
  std::cout << "Layers: " << std::endl;
  for (int i = 1; i < layers_.size(); i++) {
    std::cout << "===========" << " LinearLayer " << i << " ===========" <<
        std::endl;
    layers_[i]->PrintInfo();
  }
}
