#include <iostream>
#include "dense_layer.h"
#include "dense_net.h"
#include "softmax_layer.h"

int main() {
  DenseNet net;
  DenseLayer firstLayer(1, 3);
  Eigen::MatrixXd firstLayerWeights(3, 1);
  firstLayerWeights << 1, 1, 1;
  firstLayer.SetWeights(firstLayerWeights);
  net.AddLayer(std::make_unique<DenseLayer>(firstLayer));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(3)));
  net.PrintInfo();
  Eigen::MatrixXd input(1, 1);
  input << 1;
  Eigen::MatrixXd truth(3, 1);
  truth << 1, 0, 0;
  net.SetInputs(input);
  for (int i = 0; i < 50; i++) {
    net.Backprop(truth, 0.1);
    net.PrintInfo();
  }
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(784, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 10)));

  exit(0);
}