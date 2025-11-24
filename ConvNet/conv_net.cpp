//
// Created by Yasen on 11/22/25.
//

#include "conv_net.h"

#include "conv_input.h"

ConvNet::ConvNet() {
  conv_layers_.push_back(std::unique_ptr<ConvInput>());
}

void ConvNet::AddLayer(std::unique_ptr<ConvLayer>&& layer) {
  conv_layers_.push_back(std::move(layer));
}

void ConvNet::AddLayer(std::unique_ptr<LinearLayer>&& layer) {
  connected_layers_.push_back(std::move(layer));
}

const Eigen::VectorXf& ConvNet::Predict() {
  for (size_t i = 1; i < conv_layers_.size(); i++) {
    conv_layers_[i]->Forward(conv_layers_[i - 1]->Activation());
  }
  const Img& final_activation = conv_layers_[conv_layers_.size() - 1]->
      Activation();
  const int flattened_size = final_activation[0].cols() * final_activation[0].
                             rows();
  Eigen::VectorXf flattened(flattened_size * final_activation.size());
  for (const Eigen::MatrixXf& channel : final_activation) {
    flattened << Eigen::Map<const Eigen::VectorXf>(
        channel.data(), channel.size());
  }
  dynamic_cast<InputLayer*>(connected_layers_[0].get())->SetInputs(flattened);
  for (size_t i = 1; i < connected_layers_.size(); i++) {
    connected_layers_[i]->Forward(connected_layers_[i - 1]->Activation());
  }
  return connected_layers_[connected_layers_.size() - 1]->Activation();
}

void ConvNet::SetInputs(const Img& inputs) {
  if (ConvInput* inputLayer = dynamic_cast<ConvInput*>(conv_layers_[0].get())) {
    inputLayer->SetInputs(inputs);
  } else {
    throw std::runtime_error("LinearLayer 0 is not an InputLayer");
  }
}

float ConvNet::Backprop(const Eigen::VectorXf& expected, float learning_rate) {
  const Eigen::MatrixXf& pred = Predict();
  const Eigen::VectorXf loss = (pred - expected) * learning_rate;
  for (size_t i = connected_layers_.size() - 2; i > 0; i--) {
    connected_layers_[i]->Backward(
        connected_layers_[i - 1]->Activation(),
        connected_layers_[i + 1]->PreviousDerivative(), learning_rate);
  }
  const Eigen::VectorXf& first_connected_derivative = connected_layers_[1]->
      Activation();
  const int last_conv_channels = conv_layers_[conv_layers_.size() - 1]->
      Activation().size();
  const int channel_size = first_connected_derivative.size() /
                           last_conv_channels;
  const int new_dim = sqrt(channel_size);
  Img unflattened_derivative(last_conv_channels);
  for (int i = 0; i < last_conv_channels; i++) {
    unflattened_derivative[i].resize(new_dim, new_dim);
    unflattened_derivative[i] << first_connected_derivative.segment(
        i * channel_size, channel_size);
  }
  conv_layers_[conv_layers_.size() - 1]->Backward(
      conv_layers_[conv_layers_.size() - 2]->Activation(),
      unflattened_derivative, learning_rate);
  for (size_t i = conv_layers_.size() - 2; i > 0; i--) {
    conv_layers_[i]->Backward(conv_layers_[i - 1]->Activation(),
                              conv_layers_[i + 1]->PreviousDerivative(),
                              learning_rate);
  }
  const float cross_entropy_loss = -1 * (expected.array() * log(pred.array())).
                                   sum();
  return cross_entropy_loss;
}
