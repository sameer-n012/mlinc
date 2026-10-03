#include <math.h>

#include "kernel.h"
#include "mlc/ops.h"
#include "mlc/tensor.h"

/*
 * Internal tanh row-kernel for float32 tensors.
 */
static void mcl_tanh_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                             int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = tanhf(a[i * sa]);
    }
}

/*
 * Internal sigmoid row-kernel for float32 tensors.
 */
static void mcl_sigmoid_row_f32(const float* a, int64_t sa, float* out,
                                int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = 1.0f / (1.0f + expf(-a[i * sa]));
    }
}

/*
 * Internal ReLU row-kernel for float32 tensors.
 */
static void mcl_relu_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                             int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = fmaxf(0.0f, a[i * sa]);
    }
}

/*
 * Internal GELU row-kernel for float32 tensors.
 */
static void mcl_gelu_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                             int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        float x = a[i * sa];
        out[i * so] = 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
    }
}

/*
 * Applies the tanh activation function to the input tensor and
 * returns a new tensor
 */
mlc_tensor* mlc_tanh(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_tanh_row_f32);
}

/*
 * Applies the sigmoid activation function to the input tensor and
 * returns a new tensor
 */
mlc_tensor* mlc_sigmoid(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_sigmoid_row_f32);
}

/*
 * Applies the ReLU activation function to the input tensor and
 * returns a new tensor
 */
mlc_tensor* mlc_relu(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_relu_row_f32);
}

/*
 * Applies the ERF-based GELU activation function to the input tensor and
 * returns a new tensor
 */
mlc_tensor* mlc_gelu(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_gelu_row_f32);
}

/*
 * Applies the tanh activation function to the input tensor in-place
 */
mlc_status mlc_tanh_(mlc_tensor* tensor) {
    mlc_tensor* t = mlc_apply_unary_inplace(tensor, mcl_tanh_row_f32);
    return t == NULL ? MLC_ERROR_OUT_OF_MEMORY : MLC_SUCCESS;
}

/*
 * Applies the sigmoid activation function to the input tensor in-place
 */
mlc_status mlc_sigmoid_(mlc_tensor* tensor) {
    mlc_tensor* t = mlc_apply_unary_inplace(tensor, mcl_sigmoid_row_f32);
    return t == NULL ? MLC_ERROR_OUT_OF_MEMORY : MLC_SUCCESS;
}

/*
 * Applies the ReLU activation function to the input tensor in-place
 */
mlc_status mlc_relu_(mlc_tensor* tensor) {
    mlc_tensor* t = mlc_apply_unary_inplace(tensor, mcl_relu_row_f32);
    return t == NULL ? MLC_ERROR_OUT_OF_MEMORY : MLC_SUCCESS;
}

/*
 * Applies the ERF-based GELU activation function to the input tensor in-place
 */
mlc_status mlc_gelu_(mlc_tensor* tensor) {
    mlc_tensor* t = mlc_apply_unary_inplace(tensor, mcl_gelu_row_f32);
    return t == NULL ? MLC_ERROR_OUT_OF_MEMORY : MLC_SUCCESS;
}
