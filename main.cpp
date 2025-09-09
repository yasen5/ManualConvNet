#include <iostream>
#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "conv_layer.h"
#include "matrices.h"
#include "maxpool_layer.h"
#include "visualizer.h"

using Eigen::MatrixXf;

int main() {
    Img dummyImg(3, MatrixXf::Zero(4, 4));
    MatrixXf unpooled(4, 4);
    unpooled << 1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16;
    dummyImg.at(0) = unpooled;
    MaxpoolLayer pool(1, 1, 3);
    dummyImg = pool.activation(dummyImg);
    std::cout << dummyImg.at(0) << std::endl;
    exit(0);
}