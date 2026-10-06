#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "kernel.h"
#include "mlc/dtype.h"
#include "mlc/error.h"
#include "mlc/ops.h"
#include "mlc/tensor.h"

/*
 * Compares two float values for qsort. This function is used to sort an
 * array of floats in ascending order.
 */
static int compare_floats(const void* p1, const void* p2) {
    float f1 = *(const float*)p1;
    float f2 = *(const float*)p2;
    if (f1 < f2) return -1;
    if (f1 > f2) return 1;
    return 0;
}

/*
 * Internal function to safely convert an int64_t value to size_t. If the
 * value is negative, it returns 0. This is used to avoid issues with negative
 * sizes when calculating the total number of bytes for a tensor.
 */
static size_t int64_to_size(int64_t value) {
    if (value < 0) {
        return 0;
    }
    return (size_t)value;
}

/*
 * Pairwise summation helper for float32 arrays. This method reduces numerical
 * error.
 */
static float mlc_pairwise_sum(const float* a, int64_t sa, int64_t n) {
    if (n <= 0) {
        return 0.0f;
    }
    if (n <= 64) {
        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += a[i * sa];
        }
        return sum;
    }
    int64_t mid = n / 2;
    return mlc_pairwise_sum(a, sa, mid) +
           mlc_pairwise_sum(a + mid * sa, sa, n - mid);
}

/*
 * Normalizes the reduction dimensions, handling negative indices and ensuring
 * they are within the valid range. It also checks for duplicates and sets the
 * `dim_to_reduce` array to indicate which dimensions are to be reduced.
 */
static void mlc_normalize_reduction_dims(const int64_t* input_dims,
                                         int64_t input_ndims,
                                         int64_t tensor_ndim, int64_t* out_dims,
                                         bool* dim_to_reduce) {
    for (int64_t i = 0; i < tensor_ndim; ++i) {
        dim_to_reduce[i] = false;
    }
    for (int64_t i = 0; i < input_ndims; ++i) {
        int64_t d = input_dims[i];
        if (d < 0) {
            d += tensor_ndim;
        }
        MLC_CHECK(d >= 0 && d < tensor_ndim,
                  "Reduction dimension out of range");
        MLC_CHECK(!dim_to_reduce[d],
                  "Duplicate dimension specified for reduction");
        dim_to_reduce[d] = true;
        out_dims[i] = d;
    }
}

/*
 * Applies a reduction operation to the input tensor `a` along the specified
 * dimensions. The operation is defined by the function pointer `fn`. It
 * returns a new tensor with the reduced dimensions set to size 1.
 */
mlc_tensor* mlc_apply_reduction(const mlc_tensor* a, const int64_t* dims,
                                const int64_t ndims, mlc_reduction_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");
    MLC_CHECK(a->dtype == MLC_F32,
              "Reduction operations only support float32 tensors");

    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t normalized_dims[MLC_MAX_DIMS];

    mlc_normalize_reduction_dims(dims, ndims, a->ndim, normalized_dims,
                                 dim_to_reduce);

    int64_t reduced_shape[MLC_MAX_DIMS];
    int64_t out_ndim = a->ndim;
    for (int64_t i = 0; i < a->ndim; ++i) {
        if (dim_to_reduce[i]) {
            reduced_shape[i] = 1;
        } else {
            reduced_shape[i] = a->shape[i];
        }
    }

    mlc_tensor* out = mlc_zeros(reduced_shape, out_ndim, MLC_F32);
    if (out == NULL) return NULL;

    if (mlc_tensor_numel(out) == 0) {
        return out;
    }

    if (ndims == 1) {
        int64_t red_dim = normalized_dims[0];
        int64_t indices_out[MLC_MAX_DIMS] = {0};

        while (1) {
            int64_t offset_a = a->offset;
            int64_t offset_out = out->offset;
            for (int64_t d = 0; d < a->ndim; ++d) {
                if (d != red_dim) {
                    offset_a += indices_out[d] * a->strides[d];
                    offset_out += indices_out[d] * out->strides[d];
                }
            }

            float result = fn((const float*)a->data->data + offset_a,
                              a->strides[red_dim], a->shape[red_dim]);

            ((float*)out->data->data + offset_out)[0] = result;

            int64_t dim = a->ndim - 1;
            while (dim >= 0) {
                if (dim == red_dim) {
                    dim--;
                    continue;
                }
                indices_out[dim]++;
                if (indices_out[dim] < a->shape[dim]) {
                    break;
                }
                indices_out[dim] = 0;
                dim--;
            }
            if (dim < 0) {
                break;
            }
        }
    } else {
        if (ndims == 0) {
            mlc_tensor_free(out);
            return mlc_contiguous((mlc_tensor*)a);
        }
        mlc_tensor* current = (mlc_tensor*)a;
        for (int64_t i = 0; i < ndims; ++i) {
            mlc_tensor* next_cur =
                mlc_apply_reduction(current, &normalized_dims[i], 1, fn);
            if (current != a) {
                mlc_tensor_free(current);
            }
            if (next_cur == NULL) {
                mlc_tensor_free(out);
                return NULL;
            }
            current = next_cur;
        }
        memcpy(out->data->data, current->data->data,
               int64_to_size(mlc_tensor_numel(out)) * sizeof(float));
        if (current != a) {
            mlc_tensor_free(current);
        }
    }

    return out;
}

/*
 * Applies an index reduction operation returning MLC_I64.
 */
static mlc_tensor* mlc_apply_index_reduction(const mlc_tensor* a, int64_t dim,
                                             mlc_reduction_index_row_fn fn) {
    MLC_CHECK(a != NULL, "Input tensor is NULL");
    MLC_CHECK(fn != NULL, "Function pointer is NULL");
    MLC_CHECK(a->dtype == MLC_F32,
              "Reduction operations only support float32 tensors");

    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t normalized_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, normalized_dims,
                                 dim_to_reduce);

    int64_t reduced_shape[MLC_MAX_DIMS];
    int64_t out_ndim = a->ndim;
    for (int64_t i = 0; i < a->ndim; ++i) {
        if (dim_to_reduce[i]) {
            reduced_shape[i] = 1;
        } else {
            reduced_shape[i] = a->shape[i];
        }
    }

    mlc_tensor* out = mlc_zeros(reduced_shape, out_ndim, MLC_I64);
    if (out == NULL) return NULL;

    if (mlc_tensor_numel(out) == 0) {
        return out;
    }

    int64_t red_dim = normalized_dims[0];
    int64_t indices_out[MLC_MAX_DIMS] = {0};

    while (1) {
        int64_t offset_a = a->offset;
        int64_t offset_out = out->offset;
        for (int64_t d = 0; d < a->ndim; ++d) {
            if (d != red_dim) {
                offset_a += indices_out[d] * a->strides[d];
                offset_out += indices_out[d] * out->strides[d];
            }
        }

        int64_t result = fn((const float*)a->data->data + offset_a,
                            a->strides[red_dim], a->shape[red_dim]);

        ((int64_t*)out->data->data + offset_out)[0] = result;

        int64_t dim_idx = a->ndim - 1;
        while (dim_idx >= 0) {
            if (dim_idx == red_dim) {
                dim_idx--;
                continue;
            }
            indices_out[dim_idx]++;
            if (indices_out[dim_idx] < a->shape[dim_idx]) {
                break;
            }
            indices_out[dim_idx] = 0;
            dim_idx--;
        }
        if (dim_idx < 0) {
            break;
        }
    }

    return out;
}

/*
 * Finalizes the reduction operation by squeezing out the reduced dimensions
 * if `keepdim` is false. It returns the final output tensor.
 */
static mlc_tensor* mlc_finalize_reduction(mlc_tensor* out,
                                          const bool* dim_to_reduce,
                                          int64_t original_ndim, bool keepdim) {
    if (out == NULL) {
        return NULL;
    }
    if (keepdim) {
        return out;
    }
    mlc_tensor* current = out;
    for (int64_t i = original_ndim - 1; i >= 0; --i) {
        if (dim_to_reduce[i]) {
            mlc_tensor* squeezed = mlc_squeeze(current, i);
            mlc_tensor_free(current);
            current = squeezed;
            if (current == NULL) {
                return NULL;
            }
        }
    }
    return current;
}

/*
 * Internal summation row-kernel for float32 tensors.
 */
static float mlc_sum_row_f32(const float* a, int64_t sa, int64_t n) {
    return mlc_pairwise_sum(a, sa, n);
}

/*
 * Internal product row-kernel for float32 tensors.
 */
static float mlc_prod_row_f32(const float* a, int64_t sa, int64_t n) {
    if (n <= 0) return 1.0f;
    float prod = 1.0f;
    for (int64_t i = 0; i < n; ++i) {
        prod *= a[i * sa];
    }
    return prod;
}

/*
 * Internal mean row-kernel for float32 tensors.
 */
static float mlc_mean_row_f32(const float* a, int64_t sa, int64_t n) {
    if (n <= 0) return NAN;
    return mlc_pairwise_sum(a, sa, n) / (float)n;
}

/*
 * Internal median row-kernel for float32 tensors. This function allocates a
 * temporary tensor to store the values, sorts them, and computes the median.
 */
static float mlc_median_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    for (int64_t i = 0; i < n; ++i) {
        if (isnan(a[i * sa])) {
            return NAN;
        }
    }
    float* temp = (float*)malloc(int64_to_size(n) * sizeof(float));
    MLC_CHECK(temp != NULL, "Memory allocation failed for median computation");
    for (int64_t i = 0; i < n; ++i) {
        temp[i] = a[i * sa];
    }
    qsort(temp, int64_to_size(n), sizeof(float), compare_floats);
    float median = 0.0f;
    if (n % 2 == 1) {
        median = temp[n / 2];
    } else {
        median = temp[(n / 2) - 1];
    }
    free(temp);
    return median;
}

/*
 * Internal maximum row-kernel for float32 tensors.
 */
static float mlc_max_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float max_val = a[0];
    for (int64_t i = 0; i < n; ++i) {
        float val = a[i * sa];
        if (isnan(val) || val > max_val) {
            max_val = val;
        }
    }
    return max_val;
}

/*
 * Internal minimum row-kernel for float32 tensors.
 */
static float mlc_min_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float min_val = a[0];
    for (int64_t i = 0; i < n; ++i) {
        float val = a[i * sa];
        if (isnan(val) || val < min_val) {
            min_val = val;
        }
    }
    return min_val;
}

/*
 * Internal argmax row-kernel for float32 tensors.
 */
static int64_t mlc_argmax_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float max_val = a[0];
    int64_t best_idx = 0;
    for (int64_t i = 0; i < n; ++i) {
        float val = a[i * sa];
        if (isnan(val)) {
            best_idx = i;
            break;
        }
        if (val > max_val) {
            max_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

/*
 * Internal argmin row-kernel for float32 tensors.
 */
static int64_t mlc_argmin_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float min_val = a[0];
    int64_t best_idx = 0;
    for (int64_t i = 0; i < n; ++i) {
        float val = a[i * sa];
        if (isnan(val)) {
            best_idx = i;
            break;
        }
        if (val < min_val) {
            min_val = val;
            best_idx = i;
        }
    }
    return best_idx;
}

/*
 * Applies the summation operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_sum(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_sum_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the product operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_prod(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_prod_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the mean operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_mean(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_mean_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the median operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_median(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_median_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the maximum operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_max(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_max_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the minimum operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_min(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, 1, mlc_min_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmax operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmax(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_index_reduction(a, norm_dims[0], mlc_argmax_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmin operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmin(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(&dim, 1, a->ndim, norm_dims, dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_index_reduction(a, norm_dims[0], mlc_argmin_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the summation operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_sum_dims(const mlc_tensor* a, const int64_t* dims,
                         int64_t ndims, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, ndims, mlc_sum_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the product operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_prod_dims(const mlc_tensor* a, const int64_t* dims,
                          int64_t ndims, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_prod_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the mean operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_mean_dims(const mlc_tensor* a, const int64_t* dims,
                          int64_t ndims, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_mean_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the maximum operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_max_dims(const mlc_tensor* a, const int64_t* dims,
                         int64_t ndims, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, ndims, mlc_max_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the minimum operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_min_dims(const mlc_tensor* a, const int64_t* dims,
                         int64_t ndims, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, ndims, mlc_min_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}
