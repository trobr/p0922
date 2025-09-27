#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>
#include <algorithm>
#include "template.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define EPS 1e-12f

// utils
inline int GET_BLOCKS(const int N, const int BSIZE) {
    return (N + BSIZE - 1) / BSIZE;
}


// forward kernel
// inputs: (B, D)
// outputs: (B, out_dim) where out_dim = INCLUDE_INPUT?D:0 + NUM_FREQS*2*D
template <typename scalar_t, int D=3, bool INCLUDE_INPUT=false, int NUM_FREQS=6>
__global__ void posenc_forward_kernel(
    const scalar_t* __restrict__ inputs,
    scalar_t* __restrict__ outputs,
    const int B,
    const float* __restrict__ w_per_freq // NUM_FREQS
) {
    constexpr int OUT_DIM = (INCLUDE_INPUT ? D : 0) + NUM_FREQS * 2 * D;

    // one thread handles one (batch, dim) element across all frequencies? We'll map thread idx to element index in outputs
    int b = blockIdx.x * blockDim.x + threadIdx.x; // over B*D maybe times func
    if (b >= B) return;

    // pointer to input element value
    using Point = Vec<scalar_t, D>;
    Point x = reinterpret_cast<const Point*>(inputs)[b];
    scalar_t* local_out = outputs + b * OUT_DIM;

    int out_col = 0;
    // INCLUDE_INPUT
    if constexpr (INCLUDE_INPUT) {
        reinterpret_cast<Point*>(local_out)[0] = x;
        out_col = D;
    }

    // for each freq, write sin and cos (order sin then cos)
#pragma unroll NUM_FREQS
    for (int f = 0; f < NUM_FREQS; ++f) {
        scalar_t freq = scalar_t(1 << f); // 2^f
        scalar_t w = scalar_t(w_per_freq[f]);
        Point out1, out2;

#pragma unroll D
        for (int d = 0; d < D; ++d) {
            // compute sin/cos for x[d] at freq f
            scalar_t xdf = freq * x[d];
            // compute sin/cos with math functions
            scalar_t s = sinf(xdf);
            scalar_t c = cosf(xdf);
            // scale by w
            scalar_t vs = w * s;
            scalar_t vc = w * c;
            out1[d] = (scalar_t)vs;
            out2[d] = (scalar_t)vc;
        }
        reinterpret_cast<Point*>(local_out)[f*2+out_col] = out1;
        reinterpret_cast<Point*>(local_out)[f*2+1+out_col] = out2;
    }
}

// backward kernel
template <typename scalar_t, int D = 3, bool INCLUDE_INPUT = false, int NUM_FREQS = 6>
__global__ void posenc_backward_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ inputs,
    scalar_t* __restrict__ grad_inputs,
    const int B,
    const float* __restrict__ w_per_freq
) {
    constexpr int out_dim = (INCLUDE_INPUT ? D : 0) + NUM_FREQS * 2 * D;

    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // 每线程处理一整行 inputs[b]
    scalar_t gxi[D] = {0};  // 存每个维度的累积梯度

    int out_col = 0;
    if (INCLUDE_INPUT) {
        for (int d = 0; d < D; ++d) {
            gxi[d] += grad_out[b * out_dim + d];  // identity gradient
        }
        out_col = D;
    }

#pragma unroll NUM_FREQS
    for (int f = 0; f < NUM_FREQS; ++f) {
        scalar_t freq = scalar_t(1 << f);
        scalar_t w = scalar_t(w_per_freq[f]);
#pragma unroll D
        for (int d = 0; d < D; ++d) {
            scalar_t x = inputs[b * D + d];
            scalar_t xf = freq * x;
            scalar_t s = sinf(xf);
            scalar_t c = cosf(xf);

            scalar_t d_s = w * freq * c;
            scalar_t d_c = -w * freq * s;

            int ofs_s = out_col + f * 2 * D + d;
            int ofs_c = out_col + f * 2 * D + D + d;

            gxi[d] += d_s * grad_out[b * out_dim + ofs_s] + d_c * grad_out[b * out_dim + ofs_c];
        }
    }

    // 写回 grad_inputs
#pragma unroll D
    for (int d = 0; d < D; ++d) {
        grad_inputs[b * D + d] = gxi[d];
    }
}

// host functions
at::Tensor posenc_forward_cuda(
    const at::Tensor& inputs,
    const float iteration,
    const float kick_in_iter,
    const float full_band_iter
) {
    const int B = inputs.size(0);
    const int d = inputs.size(1);

    CHECK_CUDA(inputs);
    CHECK_CONTIGUOUS(inputs);
    if (d != 3) {
        throw std::runtime_error("posenc_forward_cuda currently only supports D=3");
    }

    // fix parameters (could be made args later)
    constexpr int NUM_FREQS = 6;
    constexpr bool INCLUDE_INPUT = false;
    constexpr int D = 3;

    // hann-like window weights (per freq index)
    float t = std::max(iteration - kick_in_iter, 0.f);
    float N = std::max(full_band_iter - kick_in_iter, 1e-6f);
    float m = (float)NUM_FREQS;
    float alpha = m * t / N;

    std::vector<float> w_h(NUM_FREQS);
    for (int i = 0; i < NUM_FREQS; ++i) {
        float diff = std::min(std::max(alpha - (float)i, 0.f), 1.f);
        w_h[i] = (1.0f - cosf(3.14159265358979323846f * diff)) / 2.0f;
    }

    // prepare device tensors
    at::Tensor w_per_freq = at::empty({NUM_FREQS}, inputs.options().dtype(at::kFloat));
    cudaMemcpy(w_per_freq.data_ptr<float>(), w_h.data(), NUM_FREQS * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int out_dim = (INCLUDE_INPUT ? D : 0) + NUM_FREQS * 2 * D;
    auto outputs = at::empty({B, out_dim}, inputs.options()).to(inputs.device());

    // kernel launch
    const int threads = 256;
    const int blocks = GET_BLOCKS(B, threads);

    TORCH_CHECK(inputs.is_cuda(), "inputs must be a CUDA tensor");
    TORCH_CHECK(outputs.is_cuda(), "outputs must be a CUDA tensor");
    TORCH_CHECK(w_per_freq.is_cuda(), "w_per_freq must be a CUDA tensor");

    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "posenc_forward_cuda", ([&] {
        posenc_forward_kernel<scalar_t, D, INCLUDE_INPUT, NUM_FREQS><<<blocks, threads>>>(
            inputs.data_ptr<scalar_t>(),
            outputs.data_ptr<scalar_t>(),
            B, 
            w_per_freq.data_ptr<float>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel launch failed");
    }

    return outputs;
}

at::Tensor posenc_backward_cuda(
    const at::Tensor& grad_out,
    const at::Tensor& inputs,
    const float iteration,
    const float kick_in_iter,
    const float full_band_iter
) {
    CHECK_CUDA(grad_out);
    CHECK_CUDA(inputs);
    CHECK_CONTIGUOUS(grad_out);
    CHECK_CONTIGUOUS(inputs);

    const int B = inputs.size(0);
    const int d = inputs.size(1);
    if (d != 3) {
        throw std::runtime_error("posenc_backward_cuda currently only supports D=3");
    }

    // fix parameters (could be made args later)
    constexpr int NUM_FREQS = 6;
    constexpr bool INCLUDE_INPUT = false;
    constexpr int D = 3;

    // Recompute freq_bands and w_per_freq on host (same as forward) - cheap
    std::vector<float> freq_bands_h(NUM_FREQS);
    for (int i = 0; i < NUM_FREQS; ++i) {
        freq_bands_h[i] = powf(2.0f, i);
    }
    float t = iteration - kick_in_iter;
    if (t < 0.f) t = 0.f;
    float N = full_band_iter - kick_in_iter;
    if (N < 1e-6f) N = 1e-6f;
    float m = (float)NUM_FREQS;
    float alpha = m * t / N;
    std::vector<float> w_h(NUM_FREQS);
    for (int i = 0; i < NUM_FREQS; ++i) {
        float diff = alpha - (float)i;
        if (diff < 0.f) diff = 0.f;
        if (diff > 1.f) diff = 1.f;
        w_h[i] = (1.0f - cosf(3.14159265358979323846f * diff)) / 2.0f;
    }

    auto options = inputs.options().dtype(at::kFloat);
    at::Tensor freq_bands = at::empty({NUM_FREQS}, options);
    at::Tensor w_per_freq = at::empty({NUM_FREQS}, options);
    // copy from host to device
    cudaMemcpy(freq_bands.data_ptr<float>(), freq_bands_h.data(), NUM_FREQS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_per_freq.data_ptr<float>(), w_h.data(), NUM_FREQS * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int out_dim = (INCLUDE_INPUT ? D : 0) + NUM_FREQS * 2 * D;
    if (!(grad_out.size(0) == B && grad_out.size(1) == out_dim)) {
        throw std::runtime_error("grad_out shape mismatch");
    }

    auto grad_inputs = at::zeros_like(inputs);

    const int threads = 256;
    const int blocks = GET_BLOCKS(B * D, threads);

    AT_DISPATCH_FLOATING_TYPES(inputs.scalar_type(), "posenc_backward_cuda", ([&] {
        posenc_backward_kernel<scalar_t, 3, false, 6><<<blocks, threads>>>(
            grad_out.data_ptr<scalar_t>(),
            inputs.data_ptr<scalar_t>(),
            grad_inputs.data_ptr<scalar_t>(),
            B,
            w_per_freq.data_ptr<float>()
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel launch failed");
    }

    return grad_inputs;
}