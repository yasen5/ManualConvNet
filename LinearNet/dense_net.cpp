//
// Created by Yasen on 9/17/25.
//

#include "dense_net.h"
#include <iostream>

#include "input_layer.h"
#include "../Common/softmax_layer.h"

DenseNet::DenseNet() {
  layers_.push_back(std::make_unique<InputLayer>());
}


const Eigen::VectorXf& DenseNet::Predict() {
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

float DenseNet::Backprop(const Eigen::VectorXf& expected,
                         const float learning_rate) {
  const Eigen::VectorXf pred = Predict();
  const Eigen::VectorXf loss_derivative = pred - expected;
  layers_[layers_.size() - 1]->Backward(
      layers_[layers_.size() - 2]->Activation(),
      loss_derivative, learning_rate);
  for (size_t i = layers_.size() - 2; i > 0; i--) {
    layers_[i]->Backward(layers_.at(i - 1)->Activation(),
                         layers_.at(i + 1)->PreviousDerivative(),
                         learning_rate);
  }
  const float cross_entropy_loss = (expected.array() * log(pred.array())).
      sum();
  return cross_entropy_loss;
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

void DenseNet::SetInputs(const Eigen::VectorXf& inputs) {
  if (InputLayer* inputLayer = dynamic_cast<InputLayer*>(layers_[0].get())) {
    inputLayer->SetInputs(inputs);
  } else {
    throw std::runtime_error("Layer 0 is not an InputLayer");
  }
}
