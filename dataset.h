#ifndef DATASET_H
#define DATASET_H

#include <Eigen/Dense>

struct ClassifiedImg {
  Eigen::MatrixXf img;
  u_int8_t category;
};

enum Data {
  TRAIN,
  VALID,
  TEST
};

class Dataset {
private:
  const std::vector<std::vector<ClassifiedImg>> images;
public:
  Dataset(const std::string &trainFile, const std::string &validFile, const std::string &testFile);
  static std::vector<ClassifiedImg> readData(const std::string &fileName);
  [[nodiscard]] std::vector<ClassifiedImg> getData(Data partition) const;
};

#endif //DATASET_H
