//
// Created by Yasen on 9/8/25.
//

#ifndef CONV_NET_H
#define CONV_NET_H
#include "layer.h"


class ConvNet {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    explicit ConvNet(std::vector<std::unique_ptr<Layer>> layers);
    int predict(const Img& input) const;
};



#endif //CONV_NET_H
