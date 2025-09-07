//
// Created by Yasen on 9/7/25.
//

#include "visualizer.h"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

void Visualizer::display(const std::string &name, const Eigen::MatrixXf &mat) {
    float minVal = mat.minCoeff();
    float maxVal = mat.maxCoeff();

    Eigen::MatrixXf normalized;
    if (maxVal > minVal) {
        normalized = ((mat.array() - minVal) / (maxVal - minVal)) * 255.0f;
    } else {
        normalized = Eigen::MatrixXf::Constant(mat.rows(), mat.cols(), 127.0f);
    }

    cv::Mat cv_image;
    eigen2cv(normalized, cv_image);

    cv::Mat cv_image_8u;
    cv_image.convertTo(cv_image_8u, CV_8UC1);

    // Display
    namedWindow(name, cv::WINDOW_NORMAL);
    cv::Mat display;
    resize(cv_image_8u, display, cv::Size(idealWindowSize, idealWindowSize), 0, 0, cv::INTER_NEAREST);
    imshow(name, display);
    resizeWindow(name, cv::Size(idealWindowSize, idealWindowSize));

    cv::waitKey(0);
    cv::destroyAllWindows();
}

// TODO display negatives as red and positives as blue
void Visualizer::displayColoredNegatives(const std::string &name, const Eigen::MatrixXf &mat) {

}

