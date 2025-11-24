//
// Created by Yasen on 11/22/25.
//

#ifndef CONV_NET_H
#define CONV_NET_H

#include "conv_layer.h"
#include "../LinearNet/input_layer.h"

class LinearLayer;

class ConvNet {
public:
  ConvNet();

  const Eigen::VectorXf& Predict();

  float Backprop(const Eigen::VectorXf& expected,
                 float learning_rate);

  void AddLayer(std::unique_ptr<LinearLayer>&& layer);

  void AddLayer(std::unique_ptr<ConvLayer>&& layer);

  void PrintInfo() const;

  void SetInputs(const Img& inputs);

private:
  std::vector<std::unique_ptr<NDLayer> > conv_layers_;
  std::vector<std::unique_ptr<LinearLayer> > connected_layers_;
};


#endif //CONV_NET_H
