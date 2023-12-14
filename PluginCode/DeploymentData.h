//
// Created by u153171 on 11/30/2023.
//

#pragma once

struct DPLData
{
    torch::Tensor latent_A;
    torch::Tensor latent_B;
    double interpolate_slider_value{0};
};

