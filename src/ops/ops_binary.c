#include <stdbool.h>

#include "kernel.h"
#include "math.h"
#include "mlc/ops.h"
#include "mlc/tensor.h"

/*
 * Checks if the input tensors `a` and `b` are suitable for a fast-path
 * binary operation. The function returns true if both tensors are contiguous,
 * have the same data type, and matching layouts.
 */
bool mlc_check_fastpath_binary(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_is_contiguous(a) && mlc_is_contiguous(b) &&
           a->dtype == b->dtype && mlc_layouts_match(a, b);
}

/*
 * Applies a binary operation to the input tensors `a` and `b`, producing a new
 * output tensor. The operation is defined by the function pointer `fn`. It
 * checks if the tensors are broadcastable, computes the output shape, and
 * applies the operation row-wise on the innermost dimension, storing the
 * results in the output tensor.
 */
mlc_tensor* mlc_apply_binary(const mlc_tensor* a, const mlc_tensor* b,
                             mlc_binary_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor a is NULL");
    MLC_CHECK(b != NULL, "Input tensor b is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");

    int64_t out_shape[MLC_MAX_DIMS];
    int64_t out_ndim;
    bool broadcastable = mlc_broadcast_shapes(a->shape, a->ndim, b->shape,
                                              b->ndim, out_shape, &out_ndim);
    MLC_CHECK(broadcastable, "Tensors are not broadcastable");

    mlc_tensor* out = mlc_empty(out_shape, out_ndim, a->dtype);
    if (out == NULL) {
        return NULL;
    }

    mlc_tensor* a_broadcasted = mlc_expand((mlc_tensor*)a, out_shape, out_ndim);
    mlc_tensor* b_broadcasted = mlc_expand((mlc_tensor*)b, out_shape, out_ndim);
    if (a_broadcasted == NULL || b_broadcasted == NULL) {
        mlc_tensor_free(out);
        mlc_tensor_free(a_broadcasted);
        mlc_tensor_free(b_broadcasted);
        return NULL;
    }

    if (mlc_tensor_numel(out) == 0) {
        mlc_tensor_free(a_broadcasted);
        mlc_tensor_free(b_broadcasted);
        return out;
    }

    if (mlc_check_fastpath_binary(a_broadcasted, b_broadcasted)) {
        fn((const float*)a_broadcasted->data->data + a_broadcasted->offset, 1,
           (const float*)b_broadcasted->data->data + b_broadcasted->offset, 1,
           (float*)out->data->data + out->offset, 1, mlc_tensor_numel(out));
        mlc_tensor_free(a_broadcasted);
        mlc_tensor_free(b_broadcasted);
        return out;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    int64_t offset_a = a_broadcasted->offset;
    int64_t offset_b = b_broadcasted->offset;
    int64_t offset_out = out->offset;
    while (1) {
        fn((const float*)a_broadcasted->data->data + offset_a,
           a_broadcasted->strides[a_broadcasted->ndim - 1],
           (const float*)b_broadcasted->data->data + offset_b,
           b_broadcasted->strides[b_broadcasted->ndim - 1],
           (float*)out->data->data + offset_out, out->strides[out->ndim - 1],
           out->shape[out->ndim - 1]);

        int64_t dim = out->ndim - 2;
        while (dim >= 0) {
            indices[dim]++;
            if (indices[dim] < out->shape[dim]) {
                break;
            }
            indices[dim] = 0;
            dim--;
        }
        if (dim < 0) {
            break;
        }

        offset_a = a_broadcasted->offset;
        offset_b = b_broadcasted->offset;
        offset_out = out->offset;
        for (int64_t d = 0; d < out->ndim - 1; ++d) {
            offset_a += indices[d] * a_broadcasted->strides[d];
            offset_b += indices[d] * b_broadcasted->strides[d];
            offset_out += indices[d] * out->strides[d];
        }
    }

    mlc_tensor_free(a_broadcasted);
    mlc_tensor_free(b_broadcasted);
    return out;
}

/*
 * Applies a binary operation to the input tensors `a` and `b` in-place,
 * modifying the values of tensor `a` directly. The operation is defined by
 * the function pointer `fn`. It checks if the tensors are broadcastable,
 * computes the output shape, and applies the operation row-wise on the
 * innermost dimension, storing the results in tensor `a`.
 */
mlc_tensor* mlc_apply_binary_inplace(mlc_tensor* a, const mlc_tensor* b,
                                     mlc_binary_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor a is NULL");
    MLC_CHECK(b != NULL, "Input tensor b is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");

    // If a and b share storage and have different layouts, create a copy of b
    // first
    mlc_tensor* b_to_free = NULL;
    if (a->data == b->data &&
        (a->offset != b->offset || !mlc_layouts_match(a, b))) {
        mlc_tensor* b_copy = mlc_clone((mlc_tensor*)b);
        if (b_copy == NULL) {
            return NULL;
        }
        b = b_copy;
        b_to_free = b_copy;
    }

    // Don't allow stride 0 in any dimension with size > 1 for in-place
    // operations
    for (int64_t i = 0; i < a->ndim; ++i) {
        MLC_CHECK(a->strides[i] != 0 || a->shape[i] <= 1,
                  "In-place operation requires the first tensor to have no "
                  "stride-0 dimensions with size > 1");
    }

    int64_t out_shape[MLC_MAX_DIMS];
    int64_t out_ndim;
    bool broadcastable = mlc_broadcast_shapes(a->shape, a->ndim, b->shape,
                                              b->ndim, out_shape, &out_ndim);
    MLC_CHECK(broadcastable, "Tensors are not broadcastable");
    MLC_CHECK(a->ndim == out_ndim,
              "In-place operation requires the first tensor to have the same "
              "number of dimensions as the output tensor");
    for (int64_t i = 0; i < a->ndim; ++i) {
        MLC_CHECK(
            a->shape[i] == out_shape[i],
            "In-place operation requires the first tensor to have the same "
            "shape as the output tensor");
    }

    mlc_tensor* b_broadcasted = mlc_expand((mlc_tensor*)b, out_shape, out_ndim);
    if (b_broadcasted == NULL) {
        mlc_tensor_free(b_to_free);
        return NULL;
    }

    if (mlc_tensor_numel(a) == 0) {
        mlc_tensor_free(b_broadcasted);
        mlc_tensor_free(b_to_free);
        return a;
    }

    if (mlc_check_fastpath_binary(a, b_broadcasted)) {
        fn((const float*)a->data->data + a->offset, 1,
           (const float*)b_broadcasted->data->data + b_broadcasted->offset, 1,
           (float*)a->data->data + a->offset, 1, mlc_tensor_numel(a));
        mlc_tensor_free(b_broadcasted);
        mlc_tensor_free(b_to_free);
        return a;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    int64_t offset_a = a->offset;
    int64_t offset_b = b_broadcasted->offset;
    while (1) {
        fn((const float*)a->data->data + offset_a, a->strides[a->ndim - 1],
           (const float*)b_broadcasted->data->data + offset_b,
           b_broadcasted->strides[b_broadcasted->ndim - 1],
           (float*)a->data->data + offset_a, a->strides[a->ndim - 1],
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
        offset_b = b_broadcasted->offset;
        for (int64_t d = 0; d < a->ndim - 1; ++d) {
            offset_a += indices[d] * a->strides[d];
            offset_b += indices[d] * b_broadcasted->strides[d];
        }
    }

    mlc_tensor_free(b_broadcasted);
    mlc_tensor_free(b_to_free);
    return a;
}

/*
 * Internal addition row-kernel for float32 tensors.
 */
static void mcl_add_row_f32(const float* a, int64_t sa, const float* b,
                            int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = a[i * sa] + b[i * sb];
    }
}

/*
 * Internal subtraction row-kernel for float32 tensors.
 */
static void mcl_sub_row_f32(const float* a, int64_t sa, const float* b,
                            int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = a[i * sa] - b[i * sb];
    }
}

/*
 * Internal multiplication row-kernel for float32 tensors.
 */
static void mcl_mul_row_f32(const float* a, int64_t sa, const float* b,
                            int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = a[i * sa] * b[i * sb];
    }
}

/*
 * Internal division row-kernel for float32 tensors.
 */
static void mcl_div_row_f32(const float* a, int64_t sa, const float* b,
                            int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = a[i * sa] / b[i * sb];
    }
}

/*
 * Internal power row-kernel for float32 tensors.
 */
static void mcl_pow_row_f32(const float* a, int64_t sa, const float* b,
                            int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = powf(a[i * sa], b[i * sb]);
    }
}

/*
 * Internal maximum row-kernel for float32 tensors.
 */
static void mcl_maximum_row_f32(const float* a, int64_t sa, const float* b,
                                int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = fmaxf(a[i * sa], b[i * sb]);
    }
}

/*
 * Internal minimum row-kernel for float32 tensors.
 */
static void mcl_minimum_row_f32(const float* a, int64_t sa, const float* b,
                                int64_t sb, float* out, int64_t so, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i * so] = fminf(a[i * sa], b[i * sb]);
    }
}

/*
 * Applies the addition operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_add(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_add_row_f32);
}

/*
 * Applies the subtraction operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_sub(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_sub_row_f32);
}

/*
 * Applies the multiplication operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_mul(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_mul_row_f32);
}

/*
 * Applies the division operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_div(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_div_row_f32);
}

/*
 * Applies the power operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_pow(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_pow_row_f32);
}

/*
 * Applies the maximum operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_maximum(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_maximum_row_f32);
}

/*
 * Applies the minimum operation to the input tensors and returns a
 * new tensor
 */
mlc_tensor* mlc_minimum(const mlc_tensor* a, const mlc_tensor* b) {
    return mlc_apply_binary(a, b, mcl_minimum_row_f32);
}

/*
 * Applies the addition operation to the input tensors in-place on `a`
 */
mlc_status mlc_add_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_add_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the subtraction operation to the input tensors in-place on `a`
 */
mlc_status mlc_sub_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_sub_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the multiplication operation to the input tensors in-place on `a`
 */
mlc_status mlc_mul_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_mul_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the division operation to the input tensors in-place on `a`
 */
mlc_status mlc_div_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_div_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the power operation to the input tensors in-place on `a`
 */
mlc_status mlc_pow_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_pow_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the maximum operation to the input tensors in-place on `a`
 */
mlc_status mlc_maximum_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_maximum_row_f32);
    return MLC_SUCCESS;
}

/*
 * Applies the minimum operation to the input tensors in-place on `a`
 */
mlc_status mlc_minimum_(mlc_tensor* a, const mlc_tensor* b) {
    mlc_apply_binary_inplace(a, b, mcl_minimum_row_f32);
    return MLC_SUCCESS;
}
