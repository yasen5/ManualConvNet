//
// Created by Yasen on 11/22/25.
//

#ifndef CONV_NET_H
#define CONV_NET_H

#include "conv_layer.h"
#include "../LinearNet/linear_layer.h"

class LinearLayer;

class ConvNet {
public:
  const Eigen::VectorXf& Predict(const Img& input);

  float Backprop(const Img& input, const Eigen::VectorXf& expected,
                 float learning_rate);

  void AddLayer(std::unique_ptr<LinearLayer>&& layer);

  void AddLayer(std::unique_ptr<ConvLayer>&& layer);

  void PrintInfo() const;

private:
  std::vector<std::unique_ptr<NDLayer> > conv_layers_;
  std::vector<std::unique_ptr<LinearLayer> > connected_layers_;
};


#endif //CONV_NET_H
