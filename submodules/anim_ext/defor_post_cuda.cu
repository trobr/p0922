// deform_post_cpp.cpp
#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

extern void launch_deform_post_forward(
    at::Tensor point,
    at::Tensor offset,
    at::Tensor scales,
    at::Tensor rotations,
    at::Tensor means3D_out,
    at::Tensor scales_out,
    at::Tensor rotations_out
);
extern void launch_deform_post_backward(
    at::Tensor grad_means3D,
    at::Tensor grad_scales_out,
    at::Tensor grad_rot_out,
    at::Tensor point,
    at::Tensor offset,
    at::Tensor scales,
    at::Tensor rotations,
    at::Tensor grad_point,
    at::Tensor grad_offset,
    at::Tensor grad_scales_in,
    at::Tensor grad_rot_in
) ;

// Note: we declared kernels above as extern for readability; but actual kernel functions are __global__ in .cu - we'll launch them here.


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_forward", &launch_deform_post_forward, "deform_post forward (CUDA)");
    m.def("launch_backward", &launch_deform_post_backward, "deform_post backward (CUDA)");
}