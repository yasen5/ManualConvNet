//
// Created by Yasen on 9/17/25.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include <vector>

#include "dense_layer.h"
#include "input_layer.h"
#include "linear_layer.h"

class DenseNet {
public:
  DenseNet();

  const Eigen::VectorXf& Predict();

  float Backprop(const Eigen::VectorXf& expected,
                 float learning_rate);

  void AddLayer(std::unique_ptr<LinearLayer>&& layer);

  void PrintInfo() const;

  void SetInputs(const Eigen::VectorXf& inputs);

private:
  std::vector<std::unique_ptr<LinearLayer> > layers_;
};


#endif //LINEAR_LAYER_H
