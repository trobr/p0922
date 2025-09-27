#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>


// 通用 Vec 模板
template<typename T, int N>
struct Vec {
    T data[N];  // N 个元素的数组

    __host__ __device__ T& operator[](int i) { return data[i]; }
    __host__ __device__ const T& operator[](int i) const { return data[i]; }
};
