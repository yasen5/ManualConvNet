#include <iostream>
#include "dense_layer.h"
#include "dense_net.h"

int main() {
  DenseNet net;
  net.AddLayer(std::make_unique<DenseLayer>(1, 1));
  net.PrintInfo();
  std::vector<Eigen::MatrixXd> inputs(3, Eigen::MatrixXd(1, 1));
  inputs[0] << 0;
  inputs[1] << 1;
  inputs[2] << 2;
  std::vector<Eigen::MatrixXd> expected_outputs(3, Eigen::MatrixXd(1, 1));
  expected_outputs[0] << 1;
  expected_outputs[1] << 3;
  expected_outputs[2] << 5;
  for (int i = 0; i < 50; i++) {
    net.SetInputs(inputs[i % 3]);
    net.Backprop(expected_outputs[i % 3], 0.1);
  }
  net.PrintInfo();
  exit(0);
}