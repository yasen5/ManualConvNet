#include <iostream>
#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "visualizer.h"

using Eigen::MatrixXf;

typedef std::vector<MatrixXf> Img;

class Matrices {
public:
    static MatrixXf crossCorrelation(const Img& image, const Img& kernels, const uint8_t stride, const uint8_t padding) {
        const uint16_t kernel_sz = kernels.at(0).rows(); // Could be cols instead, should be n x n though
        const uint16_t convolvedRows = (image.at(0).rows() + 2 * padding - kernel_sz) / stride + 1;
        const uint16_t convolvedCols = (image.at(0).cols() + 2 * padding - kernel_sz) / stride + 1;
        MatrixXf convolved = MatrixXf::Zero(convolvedRows + 2 * padding, convolvedCols + 2 * padding);
        for (int i = 0; i < kernels.size(); i++) {
            MatrixXf oneColorImg = image.at(i);
            MatrixXf paddedImg = MatrixXf::Zero(oneColorImg.rows() + 2 * padding, oneColorImg.cols() + 2 * padding);
            paddedImg.block(padding, padding, oneColorImg.rows(), oneColorImg.cols()) = oneColorImg;
            const MatrixXf& kernel = kernels.at(i);
            for (int row = 0; row < convolvedRows; row++) {
                for (int col = 0; col < convolvedCols; col++) {
                    convolved(row, col) += convolve(oneColorImg.block(row * stride, col * stride, kernel_sz, kernel_sz), kernel);
                }
            }
        }
        return convolved;
    }

    static float convolve(const MatrixXf& input, const MatrixXf& kernel) {
        if (input.sum() != 0) {
            std::cout;
        }
        if (kernel.rows() != input.rows() || kernel.cols() != input.cols()) {
            throw std::invalid_argument("Input size of (" + std::to_string(input.rows()) + ", " + std::to_string(input.cols()) + ") does not match input kernel size of (" + std::to_string(kernel.rows()) + ", " + std::to_string(kernel.cols()) + ")");
        }

        float sum = 0;
        for (int row = 0; row < kernel.rows(); row++) {
            for (int col = 0; col < kernel.cols(); col++) {
                sum += input(row, col) * kernel(row, col);
            }
        }
        return sum;
    }

    static MatrixXf maxPool(const MatrixXf& mat, const uint8_t padding, const uint8_t stride, const uint8_t kernel_sz) {
        const uint16_t convolvedRows = (mat.rows() + 2 * padding - kernel_sz) / stride + 1;
        const uint16_t convolvedCols = (mat.cols() + 2 * padding - kernel_sz) / stride + 1;
        MatrixXf pooled = MatrixXf::Zero(convolvedRows + 2 * padding, convolvedCols + 2 * padding);
        for (int row = 0; row < convolvedRows; row++) {
            for (int col = 0; col < convolvedCols; col++) {
                float greatest = -std::numeric_limits<float>::infinity();
                for (int i = 0; i < kernel_sz; i++) {
                    for (int j = 0; j < kernel_sz; j++) {
                        const float val = mat(row * stride + i, col * stride + j);
                        if (val > greatest) {
                            greatest = val;
                        }
                    }
                }
                pooled(row, col) = greatest;
            }
        }
        return pooled;
    }

    static void printImg(const Img& matrix) {
        for (const MatrixXf& channel : matrix) {
            std::cout << channel << "\n" << std::endl;
        }
        std::cout << std::endl;
    }

    static Img initKernel(const uint8_t kernel_sz) {
        std::vector<MatrixXf> img(kernel_sz, MatrixXf::Random(kernel_sz, kernel_sz));
        return img;
    }
};

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
    MatrixXf weird(3, 3);
    weird << 1, 3, 1,
              3, 10, 3,
              1,3, 1;
    MatrixXf horizontalEdgeDetector(3, 3);
    horizontalEdgeDetector << 1, 1, 1,
              0, 0, 0,
              -1, -1, -1;
    MatrixXf verticalEdgeDetector(3, 3);
    verticalEdgeDetector << -1, 0, 1,
              -1, 0, 1,
              -1, 0, 1;
    MatrixXf smoother(3, 3);
    smoother << 1.0/9.0, 1.0/9.0, 1.0/9.0,
              1.0/9.0, 1.0/9.0, 1.0/9.0,
              1.0/9.0, 1.0/9.0, 1.0/9.0;
    MatrixXf temp(5, 5);
    temp << 255, 0, 0, 255, 0,
            255, 0, 0, 255, 0,
            255, 0, 0, 255, 0,
            255, 0, 0, 255, 0,
            255, 0, 0, 255, 0;
    MatrixXf convolved = Matrices::crossCorrelation(Img{ data.getData(TRAIN).at(0).img }, Img { horizontalEdgeDetector }, 1, 0);
    Visualizer::display("Convolved", convolved);

    exit(0);
}