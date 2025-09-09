//
// Created by Yasen on 9/7/25.
//

#ifndef LAYER_H
#define LAYER_H

#include "matrices.h"

class Layer {
public:
  virtual ~Layer() = default;
  [[nodiscard]] virtual Img activation(const Img& input) const = 0;
};

#endif //LAYER_H
