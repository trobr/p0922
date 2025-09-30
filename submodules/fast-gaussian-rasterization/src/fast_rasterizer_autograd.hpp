/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rasterization_api.h"
#include <torch/torch.h>

// Autograd function for projection
class FastGSRasterize : public torch::autograd::Function<FastGSRasterize> {
public:
    static torch::autograd::tensor_list forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& means,                // [N, 3]
        const torch::Tensor& scales_raw,           // [N, 3]
        const torch::Tensor& rotations_raw,        // [N, 4]
        const torch::Tensor& opacities_raw,        // [N, 1]
        const torch::Tensor& sh_coefficients_0,    // [N, 1, 3]
        const torch::Tensor& sh_coefficients_rest, // [C, B-1, 3]
        const torch::Tensor& w2c,                  // [C, 4, 4]
        const torch::Tensor& cam_position,
        int active_sh_bases, // 1, 2, 4, 9
        int width,
        int height,
        float focal_x,
        float focal_y,
        float center_x,
        float center_y,
        float near_plane,
        float far_plane
    ); // rasterizer settings

    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::tensor_list grad_outputs);
};
