#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <Eigen/Dense>

class Visualizer {
private:
    constexpr static uint16_t idealWindowSize = 900;
public:
    static void display(const std::string& name, const Eigen::MatrixXf& mat);
    static void displayColoredNegatives(const std::string& name, const Eigen::MatrixXf& mat);
};

#endif //VISUALIZER_H
