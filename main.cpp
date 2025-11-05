#include <iostream>

#include "dataset.h"
#include "dense_layer.h"
#include "dense_net.h"
#include "softmax_layer.h"

int main() {
  const float LEARNING_RATE = 0.001;
  std::vector<ClassifiedImg> train =
      Dataset::ReadData(
          "/Users/yasen/CLionProjects/ManualConvNet/Data/train.csv", true);
  const int img_size = train[0].img.size();

  // DenseNet net;
  // net.AddLayer(
  //     std::make_unique<DenseLayer>(DenseLayer(img_size, 128)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(128, 64)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 32)));
  // net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(32, 10)));
  // net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(10)));

  DenseNet net;
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(9, 27)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(27, 9)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(9, 3)));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(3)));

  std::vector<ClassifiedImg> test = Dataset::ReadData(
      "/Users/yasen/CLionProjects/ManualConvNet/Data/test.csv", 3);

  std::vector<ClassifiedImg> fake;
  Eigen::MatrixXf mat(3, 3);
  mat <<
      1, 0, 0,
      0, 1, 0,
      0, 0, 1;
  Eigen::VectorXf flattened(9, 1);
  flattened << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Eigen::VectorXf one_hot(3);
  one_hot << 0, 1, 0;
  fake.emplace_back(ClassifiedImg(mat, flattened, 1, one_hot));

  for (int i = 0; i < 30; i++) {
    for (const auto& [img, flattened, digit, one_hot] : fake) {
      net.SetInputs(flattened);
      std::cout << "Actual: " << static_cast<int>(digit) << std::endl;
      std::cout << "Predicted: " << std::endl;
      const Eigen::VectorXf output = net.Predict();
      std::cout << "Output: " << output << std::endl;
      std::cout << "=============================" << std::endl;
      net.Backprop(one_hot, 0.1);
    }
  }

  exit(0);
}