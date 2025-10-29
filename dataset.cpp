#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

// Dataset::Dataset(const std::string& dataFolder, bool flatten): datasets_{
//     ReadData(dataFolder + "/train.csv", flatten),
//     ReadData(dataFolder + "/valid.csv", flatten),
//     ReadData(dataFolder + "/test.csv", flatten)}, img_size_(datasets_[0][0].img.size()) {
// };

std::vector<ClassifiedImg> Dataset::ReadData(const std::string& fileName,
                                             bool flatten,
                                             int max_images) {
  std::vector<ClassifiedImg> images;
  std::ifstream file(fileName);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << fileName << std::endl;
    return images;
  }
  std::string line;
  std::getline(file, line);

  int numPixelsX = 0;
  int numPixelsY = 0;

  for (std::string::size_type i = line.length() - 1; i > 0; i--) {
    if (line.at(i) == 'x') {
      for (std::string::size_type j = i + 1; j < line.length(); j++) {
        if (const char character = line.at(j); std::isdigit(character)) {
          numPixelsX *= 10;
          int digit = character - '0';
          numPixelsX += digit;
        } else {
          break;
        }
      }
      int tenthsPlace = 0;
      for (std::string::size_type j = i - 1; j > 0; j--) {
        if (const char character = line.at(j); std::isdigit(line.at(j))) {
          numPixelsY += (character - '0') * static_cast<int>(std::pow(
              10, tenthsPlace));
          tenthsPlace++;
        } else {
          break;
        }
      }
      break;
    }
  }

  int counter = 0;

  while (std::getline(file, line)) {
    counter++;
    std::stringstream ss(line);
    Eigen::MatrixXd img;
    if (flatten) {
      img.resize(numPixelsX * numPixelsY, 1);
    } else {
      img.resize(numPixelsY, numPixelsX);
    }
    std::string csvBox;
    std::getline(ss, csvBox, ',');
    const uint8_t label = csvBox.at(0) - '0';
    for (int row = 0; row < numPixelsY; row++) {
      for (int col = 0; col < numPixelsX; col++) {
        std::getline(ss, csvBox, ',');
        if (flatten) {
          img(row * numPixelsX + col) = std::stod(csvBox) / 255.0;
        } else {
          img(row, col) = std::stod(csvBox) / 255.0;
        }
      }
    }
    Eigen::MatrixXd one_hot(10, 1); // TODO get num_classes_ variable
    one_hot(label - 1, 0) = 1;
    images.emplace_back(img, label, one_hot);
    if (counter >= max_images) {
      break;
    }
  }
  std::cout << "Num images: " << images.size() << std::endl;
  return images;
}

// std::vector<ClassifiedImg> Dataset::GetData(const Data partition) const {
//   if (partition > datasets_.size()) {
//     std::cout << "Partition: " << partition <<
//         " is unavailable. This might be because the 'valid' category has not been created"
//         << std::endl;
//   }
//   return datasets_.at(partition);
// }

