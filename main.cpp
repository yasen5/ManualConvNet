#include <iostream>
#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "matrices.h"
#include "visualizer.h"

using Eigen::MatrixXf;

class ConvLayer {
private:
    std::vector<Img> kernels;
    const uint8_t kernel_sz, stride, padding;

public:
    ConvLayer(const uint8_t out_channels, const uint8_t kernel_sz, const uint8_t stride, const uint8_t padding) : kernels(out_channels), kernel_sz(kernel_sz), stride(stride), padding(padding) {
        for (int i = 0; i < out_channels; i++) {
            kernels[i] = Matrices::initKernel(kernel_sz);
        }
    }

     void info() const {
        std::cout << "Kernels: " << static_cast<int>(kernel_sz) << "\tStride: " << static_cast<int>(stride) << "\tPadding: " << static_cast<int>(padding) << std::endl;
        int kernelCounter = 0;
        for (const Img& img : kernels) {
            kernelCounter++;
            std::cout << "Kernel " << kernelCounter << " : " << std::endl;
            Matrices::printImg(img);
        }
    }

    Img activation(const std::vector<Img>& input) const {
        Img output(kernel_sz);
        for (const Img& kernel : kernels) {
            for (const Img& img : input) {
                output.push_back(Matrices::crossCorrelation(img, kernel, stride, padding));
            }
        }
        return output;
    }
};

int main() {
    Dataset data("/Users/yasen/ClionProjects/ManualConvNet/Data/train.csv", "/Users/yasen/ClionProjects/ManualConvNet/Data/literally nothing lol", "/Users/yasen/ClionProjects/ManualConvNet/Data/test.csv");
    exit(0);
}