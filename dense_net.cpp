//
// Created by Yasen on 9/17/25.
//

#include "dense_net.h"
#include <iostream>

const Eigen::MatrixXd& DenseNet::Predict(Eigen::MatrixXd input) const {
    layers[0]->Forward(input);
    for (int i = 1; i < layers.size(); i++) {
        layers[i]->Forward(layers[i-1]->Activation());
    }
    return layers[layers.size() - 1]->Activation();
}

void DenseNet::Backprop(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& input, const double learning_rate) const {
    const Eigen::MatrixXd loss_derivative = expected - Predict(input);
    layers[layers.size() - 1]->Backward(layers[layers.size() - 2]->Activation(), loss_derivative, learning_rate);
    for (size_t i = layers.size() - 2; i > 0; i--) {
        layers[i]->Backward(layers.at(i-1)->Activation(), layers.at(i+1)->PreviousDerrivative(), learning_rate);
    }
    layers[0]->Backward(input, layers[1]->PreviousDerrivative(), learning_rate);
}

void DenseNet::AddLayer(std::unique_ptr<Layer> layer) {
    layers.push_back(std::move(layer));
}

void DenseNet::PrintInfo() const {
  std::cout << "Layers: " << std::endl;
  for (int i = 0; i < layers.size(); i++) {
    std::cout << "===========" << " Layer " << i << " ===========" << std::endl;
    layers[i]->PrintInfo();
  }
}

