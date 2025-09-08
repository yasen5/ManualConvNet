#include <iostream>
#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "conv_layer.h"
#include "matrices.h"
#include "visualizer.h"

using Eigen::MatrixXf;

int main() {
    ConvLayer conv(3, 2, 3, 1, 0);
    exit(0);
}