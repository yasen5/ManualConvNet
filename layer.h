//
// Created by Yasen on 9/7/25.
//

#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <Eigen/Dense>

class Layer {
public:
    virtual ~Layer() = default;
    virtual void Forward(const Eigen::MatrixXd& input) = 0;
    virtual void Backward(const Eigen::MatrixXd& prevActivation, const Eigen::MatrixXd& nextDerivative, double learningRate) = 0;
    virtual const Eigen::MatrixXd& Activation() = 0;
    virtual const Eigen::MatrixXd& PreviousDerrivative() = 0;
    virtual void PrintInfo() const = 0;
};

#endif //LAYER_H
