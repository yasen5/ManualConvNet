//
// Created by Yasen on 9/17/25.
//

#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H
#include <vector>

#include "dense_layer.h"
#include "linear_layer.h"

class DenseNet {
public:
    const Eigen::VectorXf &Predict(const Eigen::VectorXf &input);

    float Backprop(const Eigen::VectorXf &input, const Eigen::VectorXf &expected,
                   float learning_rate);

    void AddLayer(std::unique_ptr<LinearLayer> &&layer);

    void PrintInfo() const;

private:
    std::vector<std::unique_ptr<LinearLayer> > layers_;
};


#endif //LINEAR_LAYER_H
