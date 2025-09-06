#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <fstream>
#include <sstream>

Dataset::Dataset(const std::string &trainFile, const std::string &validFile, const std::string &testFile):
    images{ readData(trainFile), readData(validFile), readData(testFile) } {};

std::vector<ClassifiedImg> Dataset::readData(const std::string &fileName) {
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
                const char character = line.at(j);
                if (std::isdigit(character)) {
                    numPixelsX *= 10;
                    numPixelsX += character - '0';
                }
                else {
                    break;
                }
            }
            int tenthsPlace = 0;
            for (std::string::size_type j = i - 1; j > 0; j--) {
                if (std::isdigit(line.at(j))) {
                    numPixelsY += (line.at(j) - '0') * static_cast<int>(std::pow(10, tenthsPlace));
                    tenthsPlace++;
                }
                else {
                    break;
                }
            }
            break;
        }
    }

    int maxForTest = 15;
    int counter = 0;

    while (std::getline(file, line)) {
        counter++;
        std::stringstream ss(line);
        Eigen::MatrixXf img;
        img.resize(numPixelsY, numPixelsX);
        std::string csvBox;
        std::getline(ss, csvBox, ',');
        uint8_t label = csvBox.at(0) - '0';
        for (int row = 0; row < numPixelsY; row++) {
            for (int col = 0; col < numPixelsX; col++) {
                std::getline(ss, csvBox, ',');
                img(row, col) = std::stof(csvBox) / 255.0f;
            }
        }
        images.push_back(ClassifiedImg(img, label));
        if (counter >= maxForTest) {
            break;
        }
    }

    cv::Mat cv_image;
    eigen2cv(images.at(0).img, cv_image);

    // imshow("Window", cv_image);
    // cv::waitKey(0);
    // cv::destroyAllWindows();

    return images;
}

std::vector<ClassifiedImg> Dataset::getData(const Data partition) const {
    if (partition > images.size()) {
        std::cout << "Partition: " << partition << " is unavailable. This might be because the 'valid' category has not been created" << std::endl;
    }
    return images.at(partition);
}

