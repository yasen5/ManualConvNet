#include <iostream>

#include "Common/constants.h"
#include "Common/dataset.h"
#include "LinearNet/dense_layer.h"
#include "LinearNet/dense_net.h"
#include "Common/softmax_layer.h"
#include "third_party/tqdm.cpp/include/tqdm/tqdm.h"
#include <iomanip>
// #include <matplot/matplot.h>

// using namespace matplot;

int main() {
  const std::vector<ClassifiedImg> read =
      Dataset::ReadData(
          "/Users/yasen/CLionProjects/ManualConvNet/Data/train.csv", 30);

  const int img_size = read[0].img.size();

  std::cout << "img_size: " << img_size << std::endl;

  DenseNet net;
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(img_size, 256)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(256, 128)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(128, 64)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 10)));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(10)));

  // std::vector<ClassifiedImg> test = Dataset::ReadData(
  //     "/Users/yasen/CLionProjects/ManualConvNet/Data/test.csv", 3);
  std::vector<ClassifiedImg> train;
  for (const ClassifiedImg& img : read) {
    if (img.digit == 9 || img.digit == 7) {
      train.push_back(img);
    }
  }
  if (train.empty()) {
    std::cout << "No 9s in the set " << std::endl;
    exit(0);
  }
  const bool verbose = false;
  std::vector<float> losses(10);
  std::vector<int> distribution(10);
  for (int i : tqdm::range(MLConstants::LinearConstants::EPOCHS)) {
    float epoch_loss = 0;
    for (const auto& [img, flattened, digit, one_hot] : train) {
      distribution[digit]++;
      net.SetInputs(flattened);
      const float curr_loss = net.Backprop(one_hot,
                                           MLConstants::LinearConstants::LEARNING_RATE);
      epoch_loss += curr_loss;
    }
    epoch_loss /= train.size();
    if (i % (MLConstants::LinearConstants::EPOCHS / 10) == 0) {
      losses[i / (MLConstants::LinearConstants::EPOCHS / 10)] = epoch_loss;
    }
  }
  std::cout << "Losses: " << std::endl;
  for (const double loss : losses) {
    std::cout << std::fixed << std::setprecision(2) << loss << ", ";
  }
  std::cout << std::endl;

  exit(0);
}
