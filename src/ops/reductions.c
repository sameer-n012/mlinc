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

    mlc_tensor* out = mlc_empty(reduced_shape, out_ndim, MLC_F32);
    if (out == NULL) {
        return NULL;
    }

    if (mlc_tensor_numel(a) == 0) {
        return out;
    }

    int64_t indices[MLC_MAX_DIMS] = {0};
    while (1) {
        int64_t dim = out_ndim - 2;
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
    }

    mlc_tensor_free(out);

    out = mlc_zeros(reduced_shape, out_ndim, MLC_F32);
    if (out == NULL) return NULL;

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
    if (keepdim) {
        return out;
    }
    // Squeeze out dimensions that were reduced (from right to left or sequentially)
    mlc_tensor* current = out;
    for (int64_t i = original_ndim - 1; i >= 0; --i) {
        if (dim_to_reduce[i]) {
            mlc_tensor* squeezed = mlc_squeeze(current, i);
            mlc_tensor_free(current);
            current = squeezed;
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
    if (n <= 0) return 0.0f;
    return mlc_pairwise_sum(a, sa, n) / (float)n;
}

/*
 * Internal median row-kernel for float32 tensors. This function allocates a
 * temporary array to store the values, sorts them, and computes the median.
 */
static float mlc_median_row_f32(const float* a, int64_t sa, int64_t n) {
    if (n <= 0) return 0.0f;
    float* temp = (float*)malloc((size_t)n * sizeof(float));
    if (temp == NULL) return 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        temp[i] = a[i * sa];
    }
    qsort(temp, (size_t)n, sizeof(float), compare_floats);
    float median = 0.0f;
    if (n % 2 == 1) {
        median = temp[n / 2];
    } else {
        median = (temp[(n / 2) - 1] + temp[n / 2]) / 2.0f;
    }
    free(temp);
    return median;
}

/*
 * Internal maximum row-kernel for float32 tensors.
 */
static float mlc_max_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float max_val = -FLT_MAX;
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
    float min_val = FLT_MAX;
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
static float mlc_argmax_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float max_val = -FLT_MAX;
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
    return (float)best_idx;
}

/*
 * Internal argmin row-kernel for float32 tensors.
 */
static float mlc_argmin_row_f32(const float* a, int64_t sa, int64_t n) {
    MLC_CHECK(n > 0, "Reduction over a dimension of size 0 is not supported");
    float min_val = FLT_MAX;
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
    return (float)best_idx;
}

/*
 * Applies the negation operation to the input tensor and returns a
 * new tensor
 */

/*
 * Applies the summation operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_sum(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_sum_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the product operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_prod(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_prod_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the mean operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_mean(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_mean_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the median operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_median(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_median_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the maximum operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_max(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_max_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the minimum operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_min(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_min_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmax operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmax(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_argmax_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmin operation to the input tensor `a` along the specified
 * dimension `dim`. If `keepdim` is true, the reduced dimension is retained
 * with size 1; otherwise, it is removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmin(const mlc_tensor* a, int64_t dim, bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS] = {false};
    int64_t norm_dim = (dim < 0) ? dim + a->ndim : dim;
    dim_to_reduce[norm_dim] = true;

    mlc_tensor* out = mlc_apply_reduction(a, &dim, 1, mlc_argmin_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the summation operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_sum_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
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
mlc_tensor* mlc_prod_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                      bool keepdim) {
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
mlc_tensor* mlc_mean_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                      bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_mean_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the median operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_median_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_median_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the maximum operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_max_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
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
mlc_tensor* mlc_min_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                     bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out = mlc_apply_reduction(a, norm_dims, ndims, mlc_min_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmax operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmax_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_argmax_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}

/*
 * Applies the argmin operation to the input tensor `a` along the specified
 * dimensions `dims`. If `keepdim` is true, the reduced dimensions are retained
 * with size 1; otherwise, they are removed. The function returns a new tensor
 * with the result of the operation.
 */
mlc_tensor* mlc_argmin_(const mlc_tensor* a, const int64_t* dims, int64_t ndims,
                        bool keepdim) {
    bool dim_to_reduce[MLC_MAX_DIMS];
    int64_t norm_dims[MLC_MAX_DIMS];
    mlc_normalize_reduction_dims(dims, ndims, a->ndim, norm_dims,
                                 dim_to_reduce);

    mlc_tensor* out =
        mlc_apply_reduction(a, norm_dims, ndims, mlc_argmin_row_f32);
    return mlc_finalize_reduction(out, dim_to_reduce, a->ndim, keepdim);
}
