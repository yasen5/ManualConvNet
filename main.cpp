#include <iostream>

#include "constants.h"
#include "dataset.h"
#include "dense_layer.h"
#include "dense_net.h"
#include "softmax_layer.h"
#include <tqdm/tqdm.h>
#include <implot/implot.h>
#include <iomanip>

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
      if (verbose) {
        std::cout << "One hot: " << one_hot << std::endl;
        std::cout << "Actual: " << static_cast<int>(digit) <<
            std::endl;
        std::cout << "Predicted: " << std::endl;
        const Eigen::VectorXf output = net.Predict();
        std::cout << "Output: " << output << std::endl;
        std::cout << "=============================" << std::endl;
      }

      const float curr_loss = net.Backprop(one_hot,
                                           MLConstants::LinearConstants::LEARNING_RATE);
      // if (i >= MLConstants::LinearConstants::EPOCHS / 2 && abs(curr_loss) > 1) {
      //   std::cout << "Pred: " << net.Predict();
      //   std::cout << "Actual: " << one_hot << std::endl;
      // }
      epoch_loss += curr_loss;
    }
    epoch_loss /= train.size();
    if (i % (MLConstants::LinearConstants::EPOCHS / 10) == 0) {
      losses[i / (MLConstants::LinearConstants::EPOCHS / 10)] = epoch_loss;
    }
  }
  // std::cout << "Distribution: " << std::endl;
  // for (const int num : distribution) {
  //   std::cout << num << ", " << std::endl;
  // }
  // sciplot::Plot2D plot;
  // sciplot::Vec x = sciplot::linspace(
  //     0.0, MLConstants::LinearConstants::EPOCHS,
  //     10);
  // plot.drawCurveWithPoints(x, losses);
  // sciplot::Figure fig{{plot}};
  // const sciplot::Canvas canv{{fig}};
  // canv.save("loss_curve_images/loss.png");
  // canv.show();
  ImGui::Begin("My Window");
  if (ImPlot::BeginPlot("My Plot")) {
    ImPlot::PlotBars("My Bar Plot", bar_data, 11);
    ImPlot::PlotLine("My Line Plot", x_data, y_data, 1000);
    ...
    ImPlot::EndPlot();
  }
  ImGui::End();
  std::cout << "Losses: " << std::endl;
  for (const double loss : losses) {
    std::cout << std::fixed << std::setprecision(2) << loss << ", ";
  }
  std::cout << std::endl;
  // for (const auto& [img, flattened, digit, one_hot] : train) {
  //   net.SetInputs(flattened);
  //   std::cout << "Actual: " << static_cast<int>(digit) << "\nOne Hot: " <<
  //       one_hot.transpose() << std::endl;
  //   std::cout << "Predicted: " << std::endl;
  //   const Eigen::VectorXf output = net.Predict();
  //   std::cout << "Output: " << output << std::endl;
  //   std::cout << "=============================" << std::endl;
  // }

  exit(0);
}
