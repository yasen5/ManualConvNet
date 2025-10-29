//
// Created by Yasen on 9/17/25.
//

#include "dense_net.h"
#include <iostream>

#include "input_layer.h"
#include <cmath>
#include "softmax_layer.h"

DenseNet::DenseNet() {
  layers_.push_back(std::make_unique<InputLayer>());
}


const Eigen::MatrixXd& DenseNet::Predict() {
  for (int i = 1; i < layers_.size(); i++) {
    layers_[i]->Forward(layers_[i - 1]->Activation());
  }
  return layers_[layers_.size() - 1]->Activation();
}

void DenseNet::Backprop(const Eigen::MatrixXd& expected,
                        const double learning_rate) {
  const Eigen::MatrixXd pred = Predict();
  Eigen::MatrixXd loss_derivative(expected.size(), 1);
  for (int i = 0; i < pred.size(); i++) {
    loss_derivative(i, 0) = expected(i, 0) * std::log(pred(i, 0)) * -1;
  }
  layers_[layers_.size() - 1]->Backward(
      layers_[layers_.size() - 2]->Activation(),
      loss_derivative, learning_rate);
  for (size_t i = layers_.size() - 2; i > 0; i--) {
    layers_[i]->Backward(layers_.at(i - 1)->Activation(),
                         layers_.at(i + 1)->PreviousDerrivative(),
                         learning_rate);
  }
}

void DenseNet::AddLayer(std::unique_ptr<Layer>&& layer) {
  layers_.push_back(std::move(layer));
}

void DenseNet::PrintInfo() const {
  std::cout << "Layers: " << std::endl;
  for (int i = 1; i < layers_.size(); i++) {
    std::cout << "===========" << " Layer " << i << " ===========" << std::endl;
    layers_[i]->PrintInfo();
  }
}

void DenseNet::SetInputs(const Eigen::MatrixXd& inputs) {
  if (auto* inputLayer = dynamic_cast<InputLayer*>(layers_[0].get())) {
    inputLayer->SetInputs(inputs);
  } else {
    throw std::runtime_error("Layer 0 is not an InputLayer");
  }
}
