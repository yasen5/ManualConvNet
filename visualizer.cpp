//
// Created by Yasen on 9/7/25.
//

#include "visualizer.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

void Visualizer::display(const std::string &name, const Eigen::MatrixXf &mat) {
    cv::Mat cv_image;
    eigen2cv(mat, cv_image);

    namedWindow(name, cv::WINDOW_NORMAL);
    cv::Mat display;
    resize(cv_image, display, cv::Size(idealWindowSize, idealWindowSize), 0, 0, cv::INTER_NEAREST);
    imshow(name, cv_image);
    resizeWindow(name, cv::Size(idealWindowSize, idealWindowSize));

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void Visualizer::displayColoredNegatives(const std::string &name, const Eigen::MatrixXf &mat) {


}

