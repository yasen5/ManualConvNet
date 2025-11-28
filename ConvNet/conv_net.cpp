//
// Created by Yasen on 11/22/25.
//

#include "conv_net.h"

void ConvNet::AddLayer(std::unique_ptr<ConvLayer>&& layer) {
  conv_layers_.push_back(std::move(layer));
}

void ConvNet::AddLayer(std::unique_ptr<LinearLayer>&& layer) {
  connected_layers_.push_back(std::move(layer));
}

const Eigen::VectorXf& ConvNet::Predict(const Img& input) {
  conv_layers_[0]->Forward(input);
  for (size_t i = 1; i < conv_layers_.size(); i++) {
    conv_layers_[i]->Forward(conv_layers_[i - 1]->Activation());
  }
  const Img& final_activation = conv_layers_[conv_layers_.size() - 1]->
      Activation();
  const Eigen::VectorXf& flattened = Matrices::Flatten(
      conv_layers_[conv_layers_.size() - 1]->Activation());
  connected_layers_[0]->Forward(flattened);
  for (size_t i = 1; i < connected_layers_.size(); i++) {
    connected_layers_[i]->Forward(connected_layers_[i - 1]->Activation());
  }
  return connected_layers_[connected_layers_.size() - 1]->Activation();
}

float ConvNet::Backprop(const Img& input, const Eigen::VectorXf& expected,
                        float learning_rate) {
  const Eigen::MatrixXf& pred = Predict(input);
  const Eigen::VectorXf loss = (pred - expected) * learning_rate;
  const float cross_entropy_loss = -1 * (expected.array() * log(pred.array())).
                                   sum();
  Eigen::VectorXf vector_derivative = loss;
  if (!connected_layers_.empty()) {
    connected_layers_[connected_layers_.size() - 1]->Backward(
        connected_layers_[connected_layers_.size() - 2]->Activation(),
        loss, learning_rate);
    for (int i = connected_layers_.size() - 2; i >= 0; i--) {
      if (i != 0) {
        connected_layers_[i]->Backward(
            connected_layers_[i - 1]->Activation(),
            connected_layers_[i + 1]->PreviousDerivative(), learning_rate);
      } else {
        connected_layers_[i]->Backward(
            Matrices::Flatten(
                conv_layers_[conv_layers_.size() - 1]->Activation()),
            connected_layers_[i + 1]->PreviousDerivative(), learning_rate);
      }
    }
    vector_derivative = connected_layers_[0]->PreviousDerivative();
  }
  const int last_conv_channels = conv_layers_[conv_layers_.size() - 1]->
      Activation().size();
  const Img unflattened_derivative = Matrices::Unflatten(
      vector_derivative, last_conv_channels);
  conv_layers_[conv_layers_.size() - 1]->Backward(
      conv_layers_[conv_layers_.size() - 2]->Activation(),
      unflattened_derivative, learning_rate);
  for (size_t i = conv_layers_.size() - 2; i > 0; i--) {
    conv_layers_[i]->Backward(conv_layers_[i - 1]->Activation(),
                              conv_layers_[i + 1]->PreviousDerivative(),
                              learning_rate);
  }
  return cross_entropy_loss;
}
