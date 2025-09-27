#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <torch/extension.h>
#include <vector>

// CUDA helpers
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

static inline int GET_BLOCKS(const int N) {
    int threads = 256;
    return (N + threads - 1) / threads;
}


// Declaration of kernels
__global__ void batch_rodrigues_kernel(const float* rot_vecs, float* out, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // each rot_vecs row has 3 floats
    const float vx = rot_vecs[idx * 3 + 0];
    const float vy = rot_vecs[idx * 3 + 1];
    const float vz = rot_vecs[idx * 3 + 2];
    // compute angle
    float angle = sqrtf(vx*vx + vy*vy + vz*vz) + 1e-8f;
    float x = vx / angle, y = vy / angle, z = vz / angle;
    float c = cosf(angle);
    float s = sinf(angle);
    float C = 1.0f - c;

    // Rodrigues formula
    // row-major 3x3
    out[idx*9 + 0] = x*x*C + c;
    out[idx*9 + 1] = x*y*C - z*s;
    out[idx*9 + 2] = x*z*C + y*s;

    out[idx*9 + 3] = y*x*C + z*s;
    out[idx*9 + 4] = y*y*C + c;
    out[idx*9 + 5] = y*z*C - x*s;

    out[idx*9 + 6] = z*x*C - y*s;
    out[idx*9 + 7] = z*y*C + x*s;
    out[idx*9 + 8] = z*z*C + c;
}

__global__ void build_transform_kernel(const float* R, const float* t, float* out, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B) return;
    // R: B x 3 x 3
    const float* Rb = R + idx*9;
    const float* tb = t + idx*3;
    float* ob = out + idx*16;
    // first 3 rows
    ob[0] = Rb[0]; ob[1] = Rb[1]; ob[2] = Rb[2]; ob[3] = tb[0];
    ob[4] = Rb[3]; ob[5] = Rb[4]; ob[6] = Rb[5]; ob[7] = tb[1];
    ob[8] = Rb[6]; ob[9] = Rb[7]; ob[10] = Rb[8]; ob[11] = tb[2];
    // last row
    ob[12] = 0.0f; ob[13] = 0.0f; ob[14] = 0.0f; ob[15] = 1.0f;
}


__global__ void build_joint_transforms_kernel(
    const float* Rptr, const float* tptr, const float* jptr,
    const int64_t* parents, float* out, int B, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int b = idx / N;
    int j = idx % N;

    const float* Rb = Rptr + (b*N + j) * 9;
    const float* jb = jptr + (b*N + j) * 3;

    float tx = jb[0], ty = jb[1], tz = jb[2];
    // rel joint handled later; here we build transform_mat(rot, rel_joint)
    float* ob = out + (b*N + j) * 16;
    ob[0] = Rb[0]; ob[1] = Rb[1]; ob[2] = Rb[2]; ob[3] = tx;
    ob[4] = Rb[3]; ob[5] = Rb[4]; ob[6] = Rb[5]; ob[7] = ty;
    ob[8] = Rb[6]; ob[9] = Rb[7]; ob[10] = Rb[8]; ob[11] = tz;
    ob[12] = 0.0f; ob[13] = 0.0f; ob[14] = 0.0f; ob[15] = 1.0f;
}

// Chaining kernel: naive parent-chain multiplication in topological order
__global__ void chain_transform_kernel(float* out, const int64_t* parents, int B, int N) {
    // We rely on parents array being small; perform chaining by iterating joints in increasing index
    // each thread handles 1 (b, j) and reads parent transform (which must be already computed)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * N) return;
    int b = idx / N;
    int j = idx % N;

    // If parent is itself (root), nothing to multiply
    int p = parents[j];
    if (p < 0 || p == j) return; // root or invalid
    // Multiply transforms[b, p] * transforms[b, j] -> transforms[b, j]
    float* Tj = out + (b*N + j) * 16;
    float* Tp = out + (b*N + p) * 16;
    // Compute Tp * Tj (4x4 matrices row-major)
    float R[16];
    for (int r=0; r<4; ++r) {
        for (int c=0; c<4; ++c) {
            float s = 0.0f;
            for (int k=0; k<4; ++k) {
                s += Tp[r*4 + k] * Tj[k*4 + c];
            }
            R[r*4 + c] = s;
        }
    }
    // copy back
    for (int i=0;i<16;++i) Tj[i] = R[i];
}

// batch_rodrigues: input Nx3 -> output Nx3x3
at::Tensor batch_rodrigues_cuda_forward(const at::Tensor& rot_vecs) {
    auto opts = rot_vecs.options();
    int N = rot_vecs.size(0);
    auto rot_mat = at::empty({N, 3, 3}, opts);
    // run kernel per-row simple implementation using thrust-style mapping via CUDA kernel
    const float* rv = rot_vecs.data_ptr<float>();
    float* out = rot_mat.data_ptr<float>();

    // Simple CPU fallback if not contiguous? For now require contiguous
    TORCH_CHECK(rot_vecs.is_contiguous(), "rot_vecs must be contiguous");
    // Launch a kernel that computes Rodrigues per element
    // We'll implement a simple CUDA kernel here inline for readability
    // But for simplicity use a lambda-style kernel within a device loop below via thrust is omitted.
    // Instead we'll copy to host for small N? No — we'll implement device kernel.

    const int nthreads = 256;
    const int nblocks = (N + nthreads - 1) / nthreads;

    // Kernel defined below
    batch_rodrigues_kernel<<<nblocks, nthreads>>>(
        rv, out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("batch_rodrigues kernel failed: ", cudaGetErrorString(err));
    }
    return rot_mat;
}

// transform_mat
at::Tensor transform_mat_cuda_forward(const at::Tensor& R, const at::Tensor& t) {
    auto opts = R.options();
    int B = R.size(0);
    auto T = at::empty({B, 4, 4}, opts);
    // Build T per-batch
    const float* Rptr = R.data_ptr<float>();
    const float* tptr = t.data_ptr<float>();
    float* out = T.data_ptr<float>();
    const int threads = 256;
    const int blocks = (B + threads - 1) / threads;
    build_transform_kernel<<<blocks, threads>>>(Rptr, tptr, out, B);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("build_transform_kernel failed: ", cudaGetErrorString(err));
    }
    return T;
}

// batch_rigid_transform
at::Tensor batch_rigid_transform_cuda_forward(
    const at::Tensor& rot_mats,
    const at::Tensor& translate,
    const at::Tensor& joints,
    const at::Tensor& parents) {

    // rot_mats: B x N x 3 x 3
    auto opts = rot_mats.options();
    int B = rot_mats.size(0);
    int N = rot_mats.size(1);
    auto transforms = at::empty({B, N, 4, 4}, opts);

    // For simplicity, build transforms per-joint per-batch on device kernel; a loop over joints
    const float* Rptr = rot_mats.data_ptr<float>();
    const float* tptr = translate.data_ptr<float>();
    const float* jptr = joints.data_ptr<float>();
    const int64_t* parents_ptr = parents.data_ptr<int64_t>();

    float* out = transforms.data_ptr<float>();

    // Launch kernel with (B*N) threads
    const int total = B * N;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    build_joint_transforms_kernel<<<blocks, threads>>>(
        Rptr, tptr, jptr, parents_ptr, out, B, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("build_joint_transforms_kernel failed: ", cudaGetErrorString(err));
    }

    // NOTE: This kernel fills transforms with per-joint transform; chaining across parents still required.
    // Because chaining across joints depends on parent indices, it is simplest and safe to do the chaining
    // on CPU if N is not huge OR launch a second kernel that uses parent indices to chain transforms.
    // For clarity, we now perform chaining on device with another kernel:
    chain_transform_kernel<<<blocks, threads>>>(out, parents_ptr, B, N);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        AT_ERROR("chain_transform_kernel failed: ", cudaGetErrorString(err));
    }

    return transforms;
}


// forward declarations of CUDA kernel helpers (implemented in kernels.cu)
at::Tensor batch_rodrigues_cuda_forward(const at::Tensor& rot_vecs);
at::Tensor transform_mat_cuda_forward(const at::Tensor& R, const at::Tensor& t);
at::Tensor batch_rigid_transform_cuda_forward(
    const at::Tensor& rot_mats,
    const at::Tensor& translate,
    const at::Tensor& joints,
    const at::Tensor& parents);

// Simple C++ wrappers used by Python
at::Tensor batch_rodrigues_forward(const at::Tensor& rot_vecs) {
    TORCH_CHECK(rot_vecs.device().is_cuda(), "rot_vecs must be CUDA tensor");
    return batch_rodrigues_cuda_forward(rot_vecs);
}

at::Tensor transform_mat_forward(const at::Tensor& R, const at::Tensor& t) {
    TORCH_CHECK(R.device().is_cuda() && t.device().is_cuda(), "inputs must be CUDA tensors");
    return transform_mat_cuda_forward(R, t);
}

at::Tensor batch_rigid_transform_forward(
    const at::Tensor& rot_mats,
    const at::Tensor& translate,
    const at::Tensor& joints,
    const at::Tensor& parents) {

    TORCH_CHECK(rot_mats.device().is_cuda() && translate.device().is_cuda() && joints.device().is_cuda(), "inputs must be CUDA tensors");
    return batch_rigid_transform_cuda_forward(rot_mats, translate, joints, parents);
}
