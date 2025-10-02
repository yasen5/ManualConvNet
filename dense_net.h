//
// Created by Yasen on 9/17/25.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include <vector>

#include "dense_layer.h"
#include "layer.h"

class DenseNet {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    DenseNet() = default;
    const Eigen::MatrixXd& Predict(Eigen::MatrixXd input) const;
    void Backprop(const Eigen::MatrixXd& expected, const Eigen::MatrixXd& input, double learning_rate) const;
    void AddLayer(std::unique_ptr<Layer> layer);
    void PrintInfo() const;
};



#endif //LINEAR_LAYER_H
