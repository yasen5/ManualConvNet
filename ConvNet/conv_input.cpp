//
// Created by Yasen on 11/22/25.
//

#include "conv_input.h"

#include <iostream>

void ConvInput::PrintInfo() const {
  std::cout << "=============== Input Layer with " << ((inputs_ != nullptr)
      ? inputs_->size()
      : " (uninitialized) ") << " channels " << std::endl;
}

const Img& ConvInput::Activation() {
  return *inputs_;
}

void ConvInput::Forward(const Img& input) {
  std::cerr << "Calling forward on an input layer" << std::endl;
}

const Img& ConvInput::PreviousDerivative() {
  std::cerr << "Calling previous derivative on an input layer" << std::endl;
}

void ConvInput::SetWeights(Img& new_weights) {
  std::cerr << "Settings weights on an input layer" << std::endl;
}

void ConvInput::Backward(const Img& prevActivation, const Img& nextDerivative,
                         float learningRate) {
  std::cerr << "Calling backprop on an input layer" << std::endl;
}

