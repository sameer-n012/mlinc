#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "kernel.h"
#include "mlc/error.h"
#include "mlc/ops.h"
#include "mlc/tensor.h"

static int compare_floats(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/*
 * Applies a reduction operation to the input tensor `a` along the specified
 * dimensions `dims`. The reduction operation is defined by the function
 * pointer `fn`. The function checks for valid inputs, creates an output
 * tensor, and applies the reduction operation row-wise on the innermost
 * dimension, storing the results in the output tensor. The output tensor has
 * the same number of dimensions as the input tensor, with the specified
 * reduction dimensions set to size 1.
 */
mlc_tensor* mlc_apply_reduction_kd(const mlc_tensor* a, const int64_t* dims,
                                   const int64_t ndims,
                                   mlc_reduction_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor is NULL");
    MLC_CHECK(dims != NULL, "Dims array is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");
    MLC_CHECK(a->dtype == MLC_F32,
              "Reduction operation only supports float32 tensors");

    int64_t out_shape[MLC_MAX_DIMS];
    int64_t out_ndim = a->ndim;
    MLC_CHECK(out_ndim >= 0,
              "Number of reduction dimensions exceeds tensor ndim");
    for (int64_t i = 0; i < a->ndim; ++i) {
        out_shape[i] = a->shape[i];
    }
    for (int64_t i = 0; i < ndims; ++i) {
        MLC_CHECK(i >= a->ndim, "Reduction dimension out of range");
        int64_t dim = dims[i];
        if (dim < 0) {
            dim += a->ndim;
        }
        out_shape[dim] = 1;
    }

    mlc_tensor* out = mlc_empty(out_shape, a->ndim, a->dtype);

    if (mlc_tensor_numel(out) == 0) {
        return out;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    int64_t offset_a = a->offset;
    int64_t offset_out = out->offset;
    while (1) {
        float red_out = fn((const float*)a->data->data + offset_a,
                           a->strides[a->ndim - 1], a->shape[a->ndim - 1]);
        memcpy((float*)out->data->data + offset_out, &red_out, sizeof(float));

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

        for (int64_t d = 0; d < a->ndim - 1; ++d) {
            offset_a += indices[d] * a->strides[d];
            offset_out += indices[d] * out->strides[d];
        }
    }

    return out;
}

/*
 * Applies a reduction operation to the input tensor `a` along the specified
 * dimension `dim`. The reduction operation is defined by the function pointer
 * `fn`. The function checks for valid inputs, creates an output tensor, and
 * applies the reduction operation row-wise on the innermost dimension,
 * storing the results in the output tensor. The output tensor has fewer
 * dimensions than the input tensor, with the specified reduction dimensions
 * removed.
 *
 */
mlc_tensor* mlc_apply_reduction(const mlc_tensor* a, const int64_t* dims,
                                const int64_t ndims, mlc_reduction_row_fn fn) {
    mlc_tensor* out = mlc_apply_reduction_kd(a, dims, ndims, fn);
    if (out == NULL) {
        return NULL;
    }

    out->ndim -= ndims;
    for (int64_t i = 0; i < out->ndim; ++i) {
        out->shape[i] = out->shape[i + ndims];
        out->strides[i] = out->strides[i + ndims];
    }

    return out;
}

static float mlc_sum_row_f32(const float* a, int64_t sa, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sum += a[i * sa];
    }
    return sum;
}

static float mlc_mean_row_f32(const float* a, int64_t sa, int64_t n) {
    float sum = mlc_sum_row_f32(a, sa, n);
    return sum / n;
}

static float mlc_median_row_f32(const float* a, int64_t sa, int64_t n) {
    float* temp = (float*)malloc(n * sizeof(float));
    for (int64_t i = 0; i < n; ++i) {
        temp[i] = a[i * sa];
    }
    qsort(temp, n, sizeof(float), compare_floats);
    float median;
    if (n % 2 == 0) {
        median = (temp[n / 2 - 1] + temp[n / 2]) / 2.0f;
    } else {
        median = temp[n / 2];
    }
    free(temp);
    return median;
}

static float mlc_max_row_f32(const float* a, int64_t sa, int64_t n) {
    float max_val = a[0];
    for (int64_t i = 1; i < n; ++i) {
        if (a[i * sa] > max_val) {
            max_val = a[i * sa];
        }
    }
    return max_val;
}

static float mlc_min_row_f32(const float* a, int64_t sa, int64_t n) {
    float min_val = a[0];
    for (int64_t i = 1; i < n; ++i) {
        if (a[i * sa] < min_val) {
            min_val = a[i * sa];
        }
    }
    return min_val;
}

static float mlc_prod_row_f32(const float* a, int64_t sa, int64_t n) {
    float prod = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        prod *= a[i * sa];
    }
    return prod;
}

static float mlc_argmax_row_f32(const float* a, int64_t sa, int64_t n) {
    int64_t argmax = 0;
    for (int64_t i = 1; i < n; ++i) {
        if (a[i * sa] > a[argmax * sa]) {
            argmax = i;
        }
    }
    return (float)argmax;
}

static float mlc_argmin_row_f32(const float* a, int64_t sa, int64_t n) {
    int64_t argmin = 0;
    for (int64_t i = 1; i < n; ++i) {
        if (a[i * sa] < a[argmin * sa]) {
            argmin = i;
        }
    }
    return (float)argmin;
}

mlc_tensor* mlc_sum(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_sum_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_sum_row_f32);
}

mlc_tensor* mlc_mean(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_mean_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_mean_row_f32);
}

mlc_tensor* mlc_median(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_median_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_median_row_f32);
}

mlc_tensor* mlc_max(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_max_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_max_row_f32);
}

mlc_tensor* mlc_min(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_min_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_min_row_f32);
}

mlc_tensor* mlc_prod(const mlc_tensor* a, int64_t dim, bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, &dim, 1, mlc_prod_row_f32)
                   : mlc_apply_reduction(a, &dim, 1, mlc_prod_row_f32);
}

mlc_tensor* mlc_argmax(const mlc_tensor* a, int64_t dim, bool keepdim) {
    mlc_tensor* t = keepdim
                        ? mlc_apply_reduction_kd(a, &dim, 1, mlc_argmax_row_f32)
                        : mlc_apply_reduction(a, &dim, 1, mlc_argmax_row_f32);

    // convert t to mlc_int64 tensor
    mlc_tensor* out = mlc_empty(t->shape, t->ndim, MLC_I64);
    if (out == NULL) {
        mlc_tensor_free(t);
        return NULL;
    }

    for (int64_t i = 0; i < mlc_tensor_numel(t); ++i) {
        ((int64_t*)out->data->data)[i] = (int64_t)((float*)t->data->data)[i];
    }
}

mlc_tensor* mlc_argmin(const mlc_tensor* a, int64_t dim, bool keepdim) {
    mlc_tensor* t = keepdim
                        ? mlc_apply_reduction_kd(a, &dim, 1, mlc_argmin_row_f32)
                        : mlc_apply_reduction(a, &dim, 1, mlc_argmin_row_f32);

    // convert t to mlc_int64 tensor
    mlc_tensor* out = mlc_empty(t->shape, t->ndim, MLC_I64);
    if (out == NULL) {
        mlc_tensor_free(t);
        return NULL;
    }

    for (int64_t i = 0; i < mlc_tensor_numel(t); ++i) {
        ((int64_t*)out->data->data)[i] = (int64_t)((float*)t->data->data)[i];
    }
}

mlc_tensor* mlc_sum_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_sum_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_sum_row_f32);
}

mlc_tensor* mlc_mean_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                      bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_mean_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_mean_row_f32);
}

mlc_tensor* mlc_median_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_median_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_median_row_f32);
}

mlc_tensor* mlc_max_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_max_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_max_row_f32);
}

mlc_tensor* mlc_min_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_min_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_min_row_f32);
}

mlc_tensor* mlc_prod_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                      bool keepdim) {
    return keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_prod_row_f32)
                   : mlc_apply_reduction(a, dims, ndims, mlc_prod_row_f32);
}

mlc_tensor* mlc_argmax_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    mlc_tensor* t =
        keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_argmax_row_f32)
                : mlc_apply_reduction(a, dims, ndims, mlc_argmax_row_f32);

    // convert t to mlc_int64 tensor
    mlc_tensor* out = mlc_empty(t->shape, t->ndim, MLC_I64);
    if (out == NULL) {
        mlc_tensor_free(t);
        return NULL;
    }

    for (int64_t i = 0; i < mlc_tensor_numel(t); ++i) {
        ((int64_t*)out->data->data)[i] = (int64_t)((float*)t->data->data)[i];
    }
}

mlc_tensor* mlc_argmin_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    mlc_tensor* t =
        keepdim ? mlc_apply_reduction_kd(a, dims, ndims, mlc_argmin_row_f32)
                : mlc_apply_reduction(a, dims, ndims, mlc_argmin_row_f32);

    // convert t to mlc_int64 tensor
    mlc_tensor* out = mlc_empty(t->shape, t->ndim, MLC_I64);
    if (out == NULL) {
        mlc_tensor_free(t);
        return NULL;
    }

    for (int64_t i = 0; i < mlc_tensor_numel(t); ++i) {
        ((int64_t*)out->data->data)[i] = (int64_t)((float*)t->data->data)[i];
    }
}
