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

  DenseNet net;
  net.AddLayer(
      std::make_unique<DenseLayer>(DenseLayer(img_size, 128)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(128, 64)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 32)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(32, 10)));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(10)));

  std::vector<ClassifiedImg> test = Dataset::ReadData(
      "/Users/yasen/CLionProjects/ManualConvNet/Data/test.csv", true, 3);

  for (int i = 0; i < 6; i++) {
    for (const ClassifiedImg& classified_img : train) {
      for (const auto& [img, digit, one_hot] : test) {
        std::cout << "img sum: " << img.sum() << std::endl;
        net.SetInputs(img);
        std::cout << "Actual: " << static_cast<int>(digit) << std::endl;
        std::cout << "Predicted: " << net.Predict() << std::endl;
        std::cout << "=============================" << std::endl;
        break;
      }
      net.SetInputs(classified_img.img);
      net.Backprop(classified_img.one_hot, LEARNING_RATE);
      // net.PrintInfo();
      break;
    }
  }

  exit(0);
}