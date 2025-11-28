#include <iostream>

#include "Common/constants.h"
#include "Common/dataset.h"
#include "LinearNet/dense_layer.h"
#include "LinearNet/dense_net.h"
#include "LinearNet/softmax_layer.h"
#include <iomanip>
#include <matplot/matplot.h>

#include "ConvNet/conv_net.h"
#include "ConvNet/visualizer.h"
#include "unordered_set"

static void trainDense() {
  const std::vector<ClassifiedImg> train =
      Dataset::ReadData(
          "/Users/yasen/CLionProjects/ManualConvNet/Data/train.csv", 30);

  const int img_size = train[0].img.size();

  std::cout << "img_size: " << img_size << std::endl;

  DenseNet net;
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(img_size, 256)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(256, 128)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(128, 64)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(64, 10)));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(10)));

  if (train.empty()) {
    std::cout << "No 9s in the set " << std::endl;
    exit(0);
  }
  std::vector<float> losses(10);
  std::vector<int> distribution(10);
  for (int i = 0; i < MLConstants::LinearConstants::EPOCHS; i++) {
    if (i % (MLConstants::LinearConstants::EPOCHS /
             MLConstants::LinearConstants::NUM_HASHTAGS) == 0) {
      std::cout << "#";
    }
    float epoch_loss = 0;
    for (const auto& [img, flattened, digit, one_hot] : train) {
      distribution[digit]++;
      const float curr_loss = net.Backprop(flattened, one_hot,
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

  const std::vector<double> x = matplot::linspace(
      0, MLConstants::LinearConstants::EPOCHS, 10);
  matplot::plot(x, losses, "-o");
  matplot::hold(matplot::on);
  matplot::show();

  exit(0);
}

static void trainConv() {
  const std::vector<ClassifiedImg> train =
      Dataset::ReadData(
          "/Users/yasen/CLionProjects/ManualConvNet/Data/train.csv", 1);

  ConvNet net;
  net.AddLayer(std::make_unique<ConvLayer>(ConvLayer(1, 3, 3, 1, 0)));
  net.AddLayer(std::make_unique<ConvLayer>(ConvLayer(3, 9, 3, 1, 0)));
  net.AddLayer(std::make_unique<ConvLayer>(ConvLayer(9, 27, 3, 1, 0)));
  net.AddLayer(std::make_unique<ConvLayer>(ConvLayer(27, 10, 3, 1, 0)));
  net.AddLayer(std::make_unique<DenseLayer>(DenseLayer(10 * 20 * 20, 10)));
  net.AddLayer(std::make_unique<SoftmaxLayer>(SoftmaxLayer(10)));
  for (int iter = 0; iter < 50; iter++) {
    std::cout << "Loss: " << net.Backprop(Img{train[0].img}, train[0].one_hot,
                                          MLConstants::LinearConstants::LEARNING_RATE)
        << std::endl;
  }
}

int main() {
  trainConv();
}