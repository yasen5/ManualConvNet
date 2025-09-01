#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXf;

typedef std::vector<MatrixXf> Img;

class Matrices {
public:
    static MatrixXf crossCorrelation(const Img& image, const Img& kernels) {
        const uint16_t kernel_sz = kernels.at(0).rows(); // Could be cols instead, should be n x n though
        const uint16_t convolvedRows = image.at(0).rows() - kernel_sz + 1;
        const uint16_t convolvedCols = image.at(0).cols() - kernel_sz + 1;
        MatrixXf convolved = MatrixXf::Zero(convolvedRows, convolvedCols);
        for (int i = 0; i < kernels.size(); i++) {
            const MatrixXf& oneColorImg = image.at(i);
            const MatrixXf& kernel = kernels.at(i);
            for (int row = 0; row < convolvedRows; row++) {
                for (int col = 0; col < convolvedCols; col++) {
                    convolved(row, col) += convolve(oneColorImg.block(row, col, kernel_sz, kernel_sz), kernel);
                }
            }
        }
        return convolved;
    }

    static float convolve(const MatrixXf& input, const MatrixXf& kernel) {
        if (kernel.rows() != input.rows() || kernel.cols() != input.cols()) {
            throw std::invalid_argument("Input size of (" + std::to_string(input.rows()) + ", " + std::to_string(input.cols()) + ") does not match input kernel size of (" + std::to_string(kernel.rows()) + ", " + std::to_string(kernel.cols()) + ")");
        }

        return kernel.cwiseProduct(input).sum();
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
                output.push_back(Matrices::crossCorrelation(img, kernel));
            }
        }
        return output;
    }
};

int main() {
    Img img(3);
    for (MatrixXf& mat : img) {
        mat.resize(6, 6);
        mat << 10, 10, 10, 0, 0, 0,
               10, 10, 10, 0, 0, 0,
               10, 10, 10, 0, 0, 0,
               10, 10, 10, 0, 0, 0,
               10, 10, 10, 0, 0, 0,
               10, 10, 10, 0, 0, 0;
    }

    Img kernel(3);
    for (MatrixXf& mat : kernel) {
        mat.resize(3, 3);
        mat << 1, 0, -1,
               1, 0, -1,
               1, 0, -1;
    }

    const MatrixXf result = Matrices::crossCorrelation(img, kernel);
    std::cout << result << std::endl;

    exit(0);
}