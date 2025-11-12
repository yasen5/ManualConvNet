#ifndef DATASET_H
#define DATASET_H

#include <Eigen/Dense>

struct ClassifiedImg {
  Eigen::MatrixXf img;
  Eigen::VectorXf flattened;
  u_int8_t digit; // TODO generalize away from MNIST digits
  Eigen::VectorXf one_hot;
};

class Dataset {
public:
  static std::vector<ClassifiedImg> ReadData(const std::string &fileName, int max_images = 0);
};

#endif //DATASET_H
