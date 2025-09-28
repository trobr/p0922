#include <torch/extension.h>

// Forward/backward entrypoints implemented in smpl_cuda.cpp
at::Tensor batch_rodrigues_forward(const at::Tensor& rot_vecs);
at::Tensor transform_mat_forward(const at::Tensor& R, const at::Tensor& t);
at::Tensor batch_rigid_transform_forward(
    const at::Tensor& rot_mats,
    const at::Tensor& translate,
    const at::Tensor& joints,
    const at::Tensor& parents);

// (Optional) expose lower-level helpers
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SMPL CUDA extension (forward kernels)";
    m.def("batch_rodrigues_forward", &batch_rodrigues_forward, "batch_rodrigues_forward (CUDA)");
    m.def("transform_mat_forward", &transform_mat_forward, "transform_mat_forward (CUDA)");
    m.def("batch_rigid_transform_forward", &batch_rigid_transform_forward, "batch_rigid_transform_forward (CUDA)");
}
