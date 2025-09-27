#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

// Forward declarations of CUDA kernels implemented in .cu
at::Tensor posenc_forward_cuda(
    const at::Tensor& inputs, // (B, D)
    const float iteration,
    const float kick_in_iter,
    const float full_band_iter
);

at::Tensor posenc_backward_cuda(
    const at::Tensor& grad_out, // (B, out_dim)
    const at::Tensor& inputs,   // (B, D)
    const float iteration,
    const float kick_in_iter,
    const float full_band_iter
);

// Python bindings
at::Tensor forward_wrapper(
    const at::Tensor& inputs,
    const double iteration,
    const double kick_in_iter,
    const double full_band_iter
) {
    TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
    return posenc_forward_cuda(inputs, (float)iteration, (float)kick_in_iter,
                               (float)full_band_iter);
}

at::Tensor backward_wrapper(
    const at::Tensor& grad_out,
    const at::Tensor& inputs,
    const double iteration,
    const double kick_in_iter,
    const double full_band_iter
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
    return posenc_backward_cuda(grad_out, inputs, (float)iteration, (float)kick_in_iter,
                                (float)full_band_iter);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_wrapper, "PosEnc forward (CUDA)");
    m.def("backward", &backward_wrapper, "PosEnc backward (CUDA)");
}