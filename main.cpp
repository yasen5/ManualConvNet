#include <iostream>
#include "dataset.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "matrices.h"
#include "visualizer.h"

using Eigen::MatrixXf;

int main() {
    Dataset data("/Users/yasen/ClionProjects/ManualConvNet/Data/train.csv", "/Users/yasen/ClionProjects/ManualConvNet/Data/literally nothing lol", "/Users/yasen/ClionProjects/ManualConvNet/Data/test.csv");
    exit(0);
}