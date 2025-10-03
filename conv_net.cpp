//
// Created by Yasen on 9/8/25.
//

#include "conv_net.h"

#include "visualizer.h"

ConvNet::ConvNet(std::vector<std::unique_ptr<Layer>> layers) : layers(std::move(layers)) {}

int ConvNet::predict(const Img &input) const {
    Img channels = input;
    for (const std::unique_ptr<Layer>& layer : layers) {
        Matrices::printImg(channels);
        channels = layer->activation(channels);
    }
    Matrices::printImg(channels);
    Visualizer::display("Processes", channels.at(0));
    return 0;
}


