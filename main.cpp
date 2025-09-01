#include <iostream>
#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::MatrixXf;
class MatrixMath {
public:
    static std::array<MatrixXf, 3> convolveRGB(const std::array<MatrixXf, 3>& image, const std::array<MatrixXf, 3>& kernels) {
        const u_int16_t kernel_sz = kernels.at(0).rows(); // Could be cols instead, should be n x n though
        const u_int16_t convolvedRows = image.at(0).rows() - kernel_sz + 1;
        const u_int16_t convolvedCols = image.at(0).cols() - kernel_sz + 1;
        std::array<Eigen::MatrixXf, 3> convolved;
        for (MatrixXf& matrix : convolved) {
            matrix.resize(convolvedRows, convolvedCols);
        }
        for (int i = 0; i < kernels.size(); i++) {
            const MatrixXf& oneColorImg = image.at(i);
            for (int row = 0; row < convolvedRows; row++) {
                for (int col = 0; col < convolvedCols; col++) {
                    convolved.at(i)(row, col) = convolve(oneColorImg.block(row, col, kernel_sz, kernel_sz), kernels.at(i));
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

    static void printImg(const std::array<MatrixXf, 3>& matrix) {
        for (const MatrixXf& channel : matrix) {
            std::cout << channel << "\n" << std::endl;
        }
        std::cout << std::endl;
    }
};

int main() {
    std::cout << "The code's been updated" << std::endl;
    std::array<MatrixXf, 3> rgbImage;
    for (int i = 0; i < 3; i++) {
        rgbImage.at(i).resize(3, 3);
        rgbImage.at(i) << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    }
    std::array<MatrixXf, 3> smootherKernel;
    for (int i = 0; i < 3; i++) {
        smootherKernel.at(i).resize(2, 2);
        smootherKernel.at(i) << 0.25, 0.25, 0.25, 0.25;
    }

    const std::array<MatrixXf, 3> smoothed = MatrixMath::convolveRGB(rgbImage, smootherKernel);

    exit(0);
}