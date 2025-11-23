//
// Created by Yasen on 11/4/25.
//

#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace MLConstants {
  class LinearConstants {
  public:
    static constexpr float LEARNING_RATE = 0.001;
    static constexpr int INPUT_SIZE = 784;
    static constexpr int EPOCHS = 1500;
    static constexpr int NUM_HASHTAGS = 180;
  };
}
#endif // CONSTANTS_H
