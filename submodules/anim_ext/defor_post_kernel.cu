// deform_post_cuda.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <math.h>

// small eps for numerical stability
#define EPS 1e-12f

// utility
inline __device__ float safe_norm4(const float *a) {
    float s = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] + a[3]*a[3];
    return sqrtf(fmaxf(s, EPS));
}

inline __device__ float safe_norm3(const float *a) {
    float s = a[0]*a[0] + a[1]*a[1] + a[2]*a[2];
    return sqrtf(fmaxf(s, EPS));
}

// Cross product of two 3-vectors a x b -> out
inline __device__ void cross3(const float *a, const float *b, float *out) {
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

// quat conversion helpers (all on device)
// wxyz -> xyzw
inline __device__ void quat_wxyz_to_xyzw(const float *in, float *out) {
    // in: [w,x,y,z] -> out: [x,y,z,w]
    out[0] = in[1];
    out[1] = in[2];
    out[2] = in[3];
    out[3] = in[0];
}

// xyzw -> wxyz
inline __device__ void quat_xyzw_to_wxyz(const float *in, float *out) {
    // in: [x,y,z,w] -> out: [w,x,y,z]
    out[0] = in[3];
    out[1] = in[0];
    out[2] = in[1];
    out[3] = in[2];
}

// quaternion product for p (xyzw) * q (xyzw) -> r (xyzw)
// uses vector/scalar split: p = (v_p, s_p)
inline __device__ void quat_product_xyzw(const float *p, const float *q, float *r) {
    // p: [x,y,z,w] where v_p = p[0..2], s_p = p[3]
    // q: [x,y,z,w]
    float v_p[3] = {p[0], p[1], p[2]};
    float v_q[3] = {q[0], q[1], q[2]};
    float s_p = p[3];
    float s_q = q[3];

    float vx1[3];
    // s_p * v_q
    vx1[0] = s_p * v_q[0];
    vx1[1] = s_p * v_q[1];
    vx1[2] = s_p * v_q[2];

    float vx2[3];
    // s_q * v_p
    vx2[0] = s_q * v_p[0];
    vx2[1] = s_q * v_p[1];
    vx2[2] = s_q * v_p[2];

    float vx3[3];
    cross3(v_p, v_q, vx3);

    r[0] = vx1[0] + vx2[0] + vx3[0];
    r[1] = vx1[1] + vx2[1] + vx3[1];
    r[2] = vx1[2] + vx2[2] + vx3[2];

    r[3] = s_p * s_q - (v_p[0]*v_q[0] + v_p[1]*v_q[1] + v_p[2]*v_q[2]);
}

// Kernel: forward fused
// Inputs:
//  point (N x 3)
//  offset (N x 10)
//  scales_in (N x 3)
//  rotations_in (N x 4)  (assumed wxyz order)
// Outputs:
//  means3D_out (N x 3), scales_out (N x 3), rotations_out (N x 4) (wxyz order)
__global__ void deform_post_forward_kernel(
    const float* __restrict__ point,
    const float* __restrict__ offset,
    const float* __restrict__ scales_in,
    const float* __restrict__ rotations_in,
    float* __restrict__ means3D_out,
    float* __restrict__ scales_out,
    float* __restrict__ rotations_out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float *p_pt = point + idx*3;
    const float *p_off = offset + idx*10;
    const float *p_scales = scales_in + idx*3;
    const float *p_rot = rotations_in + idx*4;

    // means3D = point + offset[..., :3]
    means3D_out[idx*3 + 0] = p_pt[0] + p_off[0];
    means3D_out[idx*3 + 1] = p_pt[1] + p_off[1];
    means3D_out[idx*3 + 2] = p_pt[2] + p_off[2];

    // scales = scales + offset[..., 3:6]
    scales_out[idx*3 + 0] = p_scales[0] + p_off[3];
    scales_out[idx*3 + 1] = p_scales[1] + p_off[4];
    scales_out[idx*3 + 2] = p_scales[2] + p_off[5];

    // --- quaternion path ---
    // delta_rot = offset[..., 6:10]  (length 4)
    float q1_raw[4];
    // we clone delta_rot then set w=1.0 as your python code
    q1_raw[0] = 1.0f;                 // w = 1.0
    q1_raw[1] = p_off[7];             // x
    q1_raw[2] = p_off[8];             // y
    q1_raw[3] = p_off[9];             // z

    // normalize q1_raw -> q1_norm (wxyz)
    float n1 = safe_norm4(q1_raw);
    float q1_norm[4];
    q1_norm[0] = q1_raw[0] / n1;
    q1_norm[1] = q1_raw[1] / n1;
    q1_norm[2] = q1_raw[2] / n1;
    q1_norm[3] = q1_raw[3] / n1;

    // normalize rotations input -> q2_norm (wxyz)
    float n2 = safe_norm4(p_rot);
    float q2_norm[4];
    q2_norm[0] = p_rot[0] / n2;
    q2_norm[1] = p_rot[1] / n2;
    q2_norm[2] = p_rot[2] / n2;
    q2_norm[3] = p_rot[3] / n2;

    // convert to xyzw
    float p_xyzw[4], q_xyzw[4];
    quat_wxyz_to_xyzw(q1_norm, p_xyzw);
    quat_wxyz_to_xyzw(q2_norm, q_xyzw);

    // multiply p_xyzw * q_xyzw -> r_xyzw
    float r_xyzw[4];
    quat_product_xyzw(p_xyzw, q_xyzw, r_xyzw);

    // convert r_xyzw -> r_wxyz
    float r_wxyz[4];
    quat_xyzw_to_wxyz(r_xyzw, r_wxyz);

    // normalize result
    float nr = safe_norm4(r_wxyz);
    rotations_out[idx*4 + 0] = r_wxyz[0] / nr;
    rotations_out[idx*4 + 1] = r_wxyz[1] / nr;
    rotations_out[idx*4 + 2] = r_wxyz[2] / nr;
    rotations_out[idx*4 + 3] = r_wxyz[3] / nr;
}

// Kernel: backward fused
// Inputs:
//   grad_means3D (N x 3), grad_scales_out (N x 3), grad_rot_out (N x 4)
//   original inputs: point, offset, scales_in, rotations_in
// Outputs:
//   grad_point (N x 3), grad_offset (N x 10), grad_scales_in (N x 3), grad_rot_in (N x 4)
__global__ void deform_post_backward_kernel(
    const float* __restrict__ grad_means3D,
    const float* __restrict__ grad_scales_out,
    const float* __restrict__ grad_rot_out,
    const float* __restrict__ point,
    const float* __restrict__ offset,
    const float* __restrict__ scales_in,
    const float* __restrict__ rotations_in,
    float* __restrict__ grad_point,
    float* __restrict__ grad_offset,
    float* __restrict__ grad_scales_in,
    float* __restrict__ grad_rot_in,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // pointers
    const float *p_off = offset + idx*10;
    const float *p_rot = rotations_in + idx*4;

    // zero outputs first (caller ensures zero init or we set here)
    // We'll set values directly.

    // grads for means3D:
    const float *g_mean = grad_means3D + idx*3;
    grad_point[idx*3 + 0] = g_mean[0];
    grad_point[idx*3 + 1] = g_mean[1];
    grad_point[idx*3 + 2] = g_mean[2];
    // grad offset first 3
    grad_offset[idx*10 + 0] = g_mean[0];
    grad_offset[idx*10 + 1] = g_mean[1];
    grad_offset[idx*10 + 2] = g_mean[2];

    // grads for scales
    const float *g_sc = grad_scales_out + idx*3;
    grad_scales_in[idx*3 + 0] = g_sc[0];
    grad_scales_in[idx*3 + 1] = g_sc[1];
    grad_scales_in[idx*3 + 2] = g_sc[2];
    // grad offset 3:6
    grad_offset[idx*10 + 3] = g_sc[0];
    grad_offset[idx*10 + 4] = g_sc[1];
    grad_offset[idx*10 + 5] = g_sc[2];

    // now rotations path: we need full chain:
    // forward steps summary:
    // q1_raw = [1, off[7], off[8], off[9]]  (w,x,y,z)
    // q1_norm = q1_raw / ||q1_raw||
    // q2_norm = rotations_in / ||rotations_in||
    // p_xyzw = wxyz_to_xyzw(q1_norm)
    // q_xyzw = wxyz_to_xyzw(q2_norm)
    // r_xyzw = quat_product_xyzw(p_xyzw, q_xyzw)
    // r_wxyz = xyzw_to_wxyz(r_xyzw)
    // rotations_out = r_wxyz / ||r_wxyz||
    // we have grad_rot_out (wxyz order) as incoming gradient

    const float *g_rot_out = grad_rot_out + idx*4;

    // --- recompute intermediate values (same as forward) ---
    float q1_raw[4];
    q1_raw[0] = 1.0f;
    q1_raw[1] = p_off[7];
    q1_raw[2] = p_off[8];
    q1_raw[3] = p_off[9];
    float n1 = safe_norm4(q1_raw);
    float q1_norm[4] = { q1_raw[0]/n1, q1_raw[1]/n1, q1_raw[2]/n1, q1_raw[3]/n1 };

    float n2 = safe_norm4(p_rot);
    float q2_norm[4] = { p_rot[0]/n2, p_rot[1]/n2, p_rot[2]/n2, p_rot[3]/n2 };

    float p_xyzw[4]; quat_wxyz_to_xyzw(q1_norm, p_xyzw);
    float q_xyzw[4]; quat_wxyz_to_xyzw(q2_norm, q_xyzw);

    float r_xyzw[4];
    quat_product_xyzw(p_xyzw, q_xyzw, r_xyzw);

    float r_wxyz[4];
    quat_xyzw_to_wxyz(r_xyzw, r_wxyz);

    float nr = safe_norm4(r_wxyz);
    // normalized rotations_out were r_wxyz / nr

    // --- backprop through final normalization: rotations_out = r_wxyz / nr
    // J_norm: dy/dx = (I - y y^T) / ||x||
    float y[4];
    for (int i=0;i<4;++i) y[i] = r_wxyz[i] / nr;

    // grad wrt r_wxyz
    float grad_r_wxyz[4] = {0,0,0,0};
    // grad_rot_out (incoming) is with respect to normalized output y
    // grad_r = J^T * grad_out ; but J is symmetric here -> J^T = J
    for (int i=0;i<4;++i) {
        for (int j=0;j<4;++j) {
            // (I - y y^T)/nr
            float Jij = ((i==j)?1.0f:0.0f) - y[i]*y[j];
            Jij /= nr;
            grad_r_wxyz[i] += Jij * g_rot_out[j];
        }
    }

    // grad_r_wxyz -> grad_r_xyzw via permutation (r_wxyz = [r_xyzw[3], r_xyzw[0], r_xyzw[1], r_xyzw[2]])
    // So r_xyzw = [r_wxyz[1], r_wxyz[2], r_wxyz[3], r_wxyz[0]]
    float grad_r_xyzw[4];
    grad_r_xyzw[0] = grad_r_wxyz[1];
    grad_r_xyzw[1] = grad_r_wxyz[2];
    grad_r_xyzw[2] = grad_r_wxyz[3];
    grad_r_xyzw[3] = grad_r_wxyz[0];

    // Now r_xyzw = quat_product_xyzw(p_xyzw, q_xyzw)
    // We compute gradients to p_xyzw and q_xyzw via analytic Jacobians

    // split into vector/scalar
    float v_p[3] = { p_xyzw[0], p_xyzw[1], p_xyzw[2] };
    float v_q[3] = { q_xyzw[0], q_xyzw[1], q_xyzw[2] };
    float s_p = p_xyzw[3];
    float s_q = q_xyzw[3];

    // grad incoming to r (xyzw)
    float g_r_v[3] = { grad_r_xyzw[0], grad_r_xyzw[1], grad_r_xyzw[2] };
    float g_r_s = grad_r_xyzw[3];

    // Derivatives:
    // r_v = s_p * v_q + s_q * v_p + v_p x v_q
    // r_s = s_p * s_q - v_p·v_q
    //
    // ∂r_v/∂v_p = s_q * I + [v_q]_x
    // ∂r_v/∂s_p = v_q
    // ∂r_s/∂v_p = -v_q^T
    // ∂r_s/∂s_p = s_q
    //
    // ∂r_v/∂v_q = s_p * I - [v_p]_x
    // ∂r_v/∂s_q = v_p
    // ∂r_s/∂v_q = -v_p^T
    // ∂r_s/∂s_q = s_p

    // compute grad_p_xyzw (4)
    float grad_p_v[3] = {0,0,0};
    float grad_p_s = 0.0f;

    // grad contributions from r_v
    // grad_p_v += (∂r_v/∂v_p)^T * g_r_v
    // (s_q * I + [v_q]_x)^T = s_q * I - [v_q]_x  (since cross product matrix is skew-symmetric)
    float cross_q[3] = { v_q[0], v_q[1], v_q[2] }; // will use matrix form implicitly
    // compute (s_q * I - [v_q]_x) * g_r_v
    // [v_q]_x * g_r_v = v_q x g_r_v
    float vq_x_grv[3];
    cross3(v_q, g_r_v, vq_x_grv);
    for (int i=0;i<3;++i) {
        grad_p_v[i] += s_q * g_r_v[i] - vq_x_grv[i];
    }
    // grad_p_s += (∂r_v/∂s_p)^T * g_r_v = v_q · g_r_v
    grad_p_s += v_q[0]*g_r_v[0] + v_q[1]*g_r_v[1] + v_q[2]*g_r_v[2];

    // contributions from r_s
    // grad_p_v += (∂r_s/∂v_p)^T * g_r_s = (-v_q) * g_r_s
    for (int i=0;i<3;++i) grad_p_v[i] += -v_q[i] * g_r_s;
    // grad_p_s += (∂r_s/∂s_p) * g_r_s = s_q * g_r_s
    grad_p_s += s_q * g_r_s;

    // grad to q_xyzw
    float grad_q_v[3] = {0,0,0};
    float grad_q_s = 0.0f;

    // contributions from r_v: (∂r_v/∂v_q)^T * g_r_v
    // ∂r_v/∂v_q = s_p * I + ??? Wait formula earlier: r_v = s_p v_q + s_q v_p + v_p x v_q
    // derivative wrt v_q: s_p * I + [v_p]_x *? check sign: derivative of v_p x v_q wrt v_q is -[v_p]_x? Actually
    // (v_p x v_q)_i = eps_ijk v_p_j v_q_k -> derivative w.r.t v_q gives cross product with v_p: v_p x (.) with sign?
    // After derivation, ∂(v_p x v_q)/∂v_q (applied to vector g) = - v_p x g (since cross is anti-symmetric). So
    // we'll implement as (s_p * I - [v_p]_x) * g_r_v

    float vp_x_grv[3];
    cross3(v_p, g_r_v, vp_x_grv); // v_p x g_r_v
    for (int i=0;i<3;++i) {
        grad_q_v[i] += s_p * g_r_v[i] + vp_x_grv[i] * (-1.0f); // - v_p x g_r_v
    }
    // grad_q_s += v_p · g_r_v
    grad_q_s += v_p[0]*g_r_v[0] + v_p[1]*g_r_v[1] + v_p[2]*g_r_v[2];

    // contributions from r_s:
    // grad_q_v += (∂r_s/∂v_q)^T * g_r_s = (-v_p) * g_r_s
    for (int i=0;i<3;++i) grad_q_v[i] += -v_p[i] * g_r_s;
    // grad_q_s += (∂r_s/∂s_q) * g_r_s = s_p * g_r_s
    grad_q_s += s_p * g_r_s;

    // pack grad_p_xyzw and grad_q_xyzw
    float grad_p_xyzw[4] = { grad_p_v[0], grad_p_v[1], grad_p_v[2], grad_p_s };
    float grad_q_xyzw[4] = { grad_q_v[0], grad_q_v[1], grad_q_v[2], grad_q_s };

    // Now map grad_p_xyzw back through p_xyzw = wxyz_to_xyzw(q1_norm)
    // q1_norm (w,x,y,z) -> p_xyzw = [x,y,z,w]
    // so grad_q1_norm (wxyz) gets:
    // grad_q1_norm[0] (w) += grad_p_xyzw[3]
    // grad_q1_norm[1] (x) += grad_p_xyzw[0]
    // grad_q1_norm[2] (y) += grad_p_xyzw[1]
    // grad_q1_norm[3] (z) += grad_p_xyzw[2]
    float grad_q1_norm[4] = { grad_p_xyzw[3], grad_p_xyzw[0], grad_p_xyzw[1], grad_p_xyzw[2] };

    // Map grad_q_xyzw back through q_xyzw = wxyz_to_xyzw(q2_norm)
    float grad_q2_norm[4] = { grad_q_xyzw[3], grad_q_xyzw[0], grad_q_xyzw[1], grad_q_xyzw[2] };

    // Now backprop through normalization for q1_norm and q2_norm:
    // q1_norm = q1_raw / n1  where q1_raw = [1, off7, off8, off9]
    // dy/dx = (I - y y^T)/||x||
    // compute grad_q1_raw
    float grad_q1_raw[4] = {0,0,0,0};
    // J_q1 = (I - q1_norm q1_norm^T)/n1
    for (int i=0;i<4;++i) {
        for (int j=0;j<4;++j) {
            float Jij = ((i==j)?1.0f:0.0f) - q1_norm[i]*q1_norm[j];
            Jij /= n1;
            grad_q1_raw[i] += Jij * grad_q1_norm[j];
        }
    }
    // BUT q1_raw[0] = 1.0 (constant) so gradient to it should be discarded (zero)
    grad_q1_raw[0] = 0.0f;

    // q1_raw components [1]..[3] correspond to offset[7], offset[8], offset[9]
    grad_offset[idx*10 + 6] = 0.0f; // offset[6] corresponds to delta_rot[0] but it was overwritten -> no grad
    grad_offset[idx*10 + 7] = grad_q1_raw[1];
    grad_offset[idx*10 + 8] = grad_q1_raw[2];
    grad_offset[idx*10 + 9] = grad_q1_raw[3];

    // For completeness, set any other offset grads (if not set earlier) to their computed values (some set above)
    // Note: offset[0..5] already assigned above; offset[6] explicitly zero.

    // Now grad for rotations input: q2_norm = rotations_in / n2
    float grad_rot_raw[4] = {0,0,0,0};
    for (int i=0;i<4;++i) {
        for (int j=0;j<4;++j) {
            float Jij = ((i==j)?1.0f:0.0f) - q2_norm[i]*q2_norm[j];
            Jij /= n2;
            grad_rot_raw[i] += Jij * grad_q2_norm[j];
        }
    }
    // write to grad_rot_in
    grad_rot_in[idx*4 + 0] = grad_rot_raw[0];
    grad_rot_in[idx*4 + 1] = grad_rot_raw[1];
    grad_rot_in[idx*4 + 2] = grad_rot_raw[2];
    grad_rot_in[idx*4 + 3] = grad_rot_raw[3];
}

void launch_deform_post_forward(
    at::Tensor point,
    at::Tensor offset,
    at::Tensor scales,
    at::Tensor rotations,
    at::Tensor means3D_out,
    at::Tensor scales_out,
    at::Tensor rotations_out
) {
    const int N = point.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    // raw pointers (assume contiguous & float)
    const float* p_point = point.data_ptr<float>();
    const float* p_offset = offset.data_ptr<float>();
    const float* p_scales = scales.data_ptr<float>();
    const float* p_rot = rotations.data_ptr<float>();

    float* p_means = means3D_out.data_ptr<float>();
    float* p_scales_out = scales_out.data_ptr<float>();
    float* p_rots_out = rotations_out.data_ptr<float>();

    // launch kernel (we define kernel name exactly from .cu)
    // using the kernel name in .cu
    deform_post_forward_kernel<<<blocks, threads>>>(
        p_point, p_offset, p_scales, p_rot, p_means, p_scales_out, p_rots_out, N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA forward kernel error: %s\n", cudaGetErrorString(err));
    }
}

void launch_deform_post_backward(
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
) {
    const int N = point.size(0);
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    const float* p_gm = grad_means3D.data_ptr<float>();
    const float* p_gs = grad_scales_out.data_ptr<float>();
    const float* p_gr = grad_rot_out.data_ptr<float>();

    const float* p_point = point.data_ptr<float>();
    const float* p_off = offset.data_ptr<float>();
    const float* p_scales = scales.data_ptr<float>();
    const float* p_rots = rotations.data_ptr<float>();

    float* p_gp = grad_point.data_ptr<float>();
    float* p_goff = grad_offset.data_ptr<float>();
    float* p_gs_in = grad_scales_in.data_ptr<float>();
    float* p_gr_in = grad_rot_in.data_ptr<float>();

    deform_post_backward_kernel<<<blocks, threads>>>(
        p_gm, p_gs, p_gr, p_point, p_off, p_scales, p_rots, p_gp, p_goff, p_gs_in, p_gr_in, N
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA backward kernel error: %s\n", cudaGetErrorString(err));
    }
}
