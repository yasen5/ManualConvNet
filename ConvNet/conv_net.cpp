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

Eigen::VectorXf ConvNet::ConvToLinear(const Img& convImg) {
  const int flattened_size = convImg[0].cols() * convImg[0].
                             rows();
  Eigen::VectorXf flattened(flattened_size * convImg.size());
  int offset = 0;
  for (const Eigen::MatrixXf& channel : convImg) {
    flattened.segment(offset, channel.size()) = Eigen::Map<const
      Eigen::VectorXf>(
        channel.data(), channel.size());
    offset += channel.size();
  }
  return flattened;
}

Img ConvNet::LinearToConv(const Eigen::VectorXf& vec,
                          const int channels) {
  const int channel_size = vec.size() / channels;
  const int new_dim = sqrt(channel_size);
  Img unflattened(channels);
  for (int i = 0; i < channels; i++) {
    unflattened[i] = Eigen::Map<const
      Eigen::MatrixXf>(vec.data() + i * channel_size, new_dim, new_dim);
  }
  return unflattened;
}

const Eigen::VectorXf& ConvNet::Predict(const Img& input) {
  conv_layers_[0]->Forward(input);
  for (size_t i = 1; i < conv_layers_.size(); i++) {
    conv_layers_[i]->Forward(conv_layers_[i - 1]->Activation());
  }
  const Img& final_activation = conv_layers_[conv_layers_.size() - 1]->
      Activation();
  const Eigen::VectorXf& flattened = ConvToLinear(
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
            ConvToLinear(conv_layers_[conv_layers_.size() - 1]->Activation()),
            connected_layers_[i + 1]->PreviousDerivative(), learning_rate);
      }
    }
    vector_derivative = connected_layers_[0]->PreviousDerivative();
  }
  const int last_conv_channels = conv_layers_[conv_layers_.size() - 1]->
      Activation().size();
  const Img unflattened_derivative = LinearToConv(
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
