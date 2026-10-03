#include "kernel.h"
#include "math.h"
#include "mlc/ops.h"
#include "mlc/tensor.h"

/*
 * Checks if the input tensor `a` is suitable for a fast-path
 * unary operation. The function returns true if the tensor is contiguous.
 */
bool mlc_check_fastpath_unary(const mlc_tensor* a) {
    return mlc_is_contiguous(a);
}

/*
 * Applies a unary operation to the input tensor `a`, producing a new output
 * tensor. The operation is defined by the function pointer `fn`. It walks
 * through the tensor dimensions and applies the operation row-wise on the
 * innermost dimension, storing the results in the output tensor.
 */
mlc_tensor* mlc_apply_unary(const mlc_tensor* a, mlc_unary_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");

    mlc_tensor* out = mlc_empty(a->shape, a->ndim, a->dtype);
    if (out == NULL) {
        return NULL;
    }

    if (mlc_check_fastpath_unary(a)) {
        fn((const float*)a->data->data + a->offset, 1,
           (float*)out->data->data + out->offset, 1, mlc_tensor_numel(a));
        return out;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    int64_t offset_a = a->offset;
    int64_t offset_out = out->offset;
    while (1) {
        fn((const float*)a->data->data + offset_a, a->strides[a->ndim - 1],
           (float*)out->data->data + offset_out, out->strides[out->ndim - 1],
           a->shape[a->ndim - 1]);

        int64_t dim = a->ndim - 2;
        while (dim >= 0) {
            indices[dim]++;
            if (indices[dim] < a->shape[dim]) {
                break;
            }
            indices[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;
        }

        offset_a = a->offset;
        offset_out = out->offset;
        for (int64_t d = 0; d < a->ndim - 1; ++d) {
            offset_a += indices[d] * a->strides[d];
            offset_out += indices[d] * out->strides[d];
        }
    }

    return out;
}

/*
 * Applies a unary operation to the input tensor `a` in-place, modifying its
 * values directly. The operation is defined by the function pointer `fn`. It
 * walks through the tensor dimensions and applies the operation row-wise on
 * the innermost dimension.
 */
mlc_tensor* mlc_apply_unary_inplace(mlc_tensor* a, mlc_unary_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");

    if (mlc_check_fastpath_unary(a)) {
        fn((const float*)a->data->data + a->offset, 1,
           (float*)a->data->data + a->offset, 1, mlc_tensor_numel(a));
        return a;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    int64_t offset = a->offset;
    while (1) {
        fn((const float*)a->data->data + offset, a->strides[a->ndim - 1],
           (float*)a->data->data + offset, a->strides[a->ndim - 1],
           a->shape[a->ndim - 1]);

        int64_t dim = a->ndim - 2;
        while (dim >= 0) {
            indices[dim]++;
            if (indices[dim] < a->shape[dim]) {
                break;
            }
            indices[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;
        }

        offset = a->offset;
        for (int64_t d = 0; d < a->ndim - 1; ++d) {
            offset += indices[d] * a->strides[d];
        }
    }

    return a;
}

/*
 * Internal negation row-kernel for float32 tensors.
 */
static void mcl_neg_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                            int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = -a[i * sa];
    }
}

/*
 * Internal absolute value row-kernel for float32 tensors.
 */
static void mcl_abs_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                            int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = fabsf(a[i * sa]);
    }
}

/*
 * Internal exponential row-kernel for float32 tensors.
 */
static void mcl_exp_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                            int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = expf(a[i * sa]);
    }
}

/*
 * Internal logarithm row-kernel for float32 tensors.
 */
static void mcl_log_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                            int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = logf(a[i * sa]);
    }
}

/*
 * Internal square root row-kernel for float32 tensors.
 */
static void mcl_sqrt_row_f32(const float* a, int64_t sa, float* out, int64_t so,
                             int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = sqrtf(a[i * sa]);
    }
}

/*
 * Applies the negation operation to the input tensor and returns a
 * new tensor
 */
mlc_tensor* mlc_neg(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_neg_row_f32);
}

/*
 * Applies the absolute value operation to the input tensor and returns a
 * new tensor
 */
mlc_tensor* mlc_abs(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_abs_row_f32);
}

/*
 * Applies the exponential operation to the input tensor and returns a
 * new tensor
 */
mlc_tensor* mlc_exp(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_exp_row_f32);
}

/*
 * Applies the logarithm operation to the input tensor and returns a
 * new tensor
 */
mlc_tensor* mlc_log(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_log_row_f32);
}

/*
 * Applies the square root operation to the input tensor and returns a
 * new tensor
 */
mlc_tensor* mlc_sqrt(const mlc_tensor* tensor) {
    return mlc_apply_unary(tensor, mcl_sqrt_row_f32);
}

/*
 * Applies the negation operation to the input tensor in-place
 */
mlc_status mlc_neg_(mlc_tensor* tensor) {
    mlc_apply_unary_inplace(tensor, mcl_neg_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the absolute value operation to the input tensor in-place
 */
mlc_status mlc_abs_(mlc_tensor* tensor) {
    mlc_apply_unary_inplace(tensor, mcl_abs_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the exponential operation to the input tensor in-place
 */
mlc_status mlc_exp_(mlc_tensor* tensor) {
    mlc_apply_unary_inplace(tensor, mcl_exp_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the logarithm operation to the input tensor in-place
 */
mlc_status mlc_log_(mlc_tensor* tensor) {
    mlc_apply_unary_inplace(tensor, mcl_log_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the square root operation to the input tensor in-place
 */
mlc_status mlc_sqrt_(mlc_tensor* tensor) {
    mlc_apply_unary_inplace(tensor, mcl_sqrt_row_f32);
    return MLC_SUCCESS;
}
