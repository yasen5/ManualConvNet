#ifndef DATASET_H
#define DATASET_H

#include <Eigen/Dense>

struct ClassifiedImg {
  Eigen::MatrixXd img;
  u_int8_t digit; // TODO generalize away from MNIST digits
  Eigen::MatrixXd one_hot;
};

// enum Data {
//   TRAIN,
//   VALID,
//   TEST
// };

class Dataset {
private:
  // const std::vector<std::vector<ClassifiedImg> > datasets_;
  // const int img_size_;

public:
  // Dataset(const std::string &dataFolder, bool flatten);

  static std::vector<ClassifiedImg> ReadData(const std::string &fileName, bool flatten, int max_images = INFINITY);

  // std::vector<ClassifiedImg> GetData(Data partition) const;

  // constexpr int ImgSize() const {
  //   return img_size_;
  // }
};

#endif //DATASET_H
