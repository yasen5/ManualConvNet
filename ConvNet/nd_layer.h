//
// Created by Yasen on 11/22/25.
//

#ifndef ND_LAYER_H
#define ND_LAYER_H

#include <Eigen/Dense>

#include "matrices.h"

class NDLayer {
public:
  virtual ~NDLayer() = default;

  virtual void Forward(const Img& input) = 0;

  virtual void Backward(const Img& prevActivation,
                        const Img& nextDerivative,
                        float learningRate) = 0;

  virtual const Img& Activation() = 0;

  virtual const Img& PreviousDerivative() = 0;

  virtual void SetWeights(std::vector<Img>& new_weights) = 0;

  virtual void PrintInfo() const = 0;
};

#endif //ND_LAYER_H
