//
// Created by Yasen on 9/17/25.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include <vector>

#include "dense_layer.h"
#include "input_layer.h"
#include "layer.h"

class DenseNet {
private:
  std::vector<std::unique_ptr<Layer> > layers_;

public:
  DenseNet();

  const Eigen::MatrixXd& Predict();

  void Backprop(const Eigen::MatrixXd& expected,
                double learning_rate);

  void AddLayer(std::unique_ptr<Layer>&& layer);

  void PrintInfo() const;

  void SetInputs(const Eigen::MatrixXd& inputs);
};


#endif //LINEAR_LAYER_H
