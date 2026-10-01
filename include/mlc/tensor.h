#ifndef MLC_TENSOR_H
#define MLC_TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

#include "dtype.h"
#include "storage.h"
#include "mlc/rng.h"

#define MLC_MAX_DIMS 8

/*
 * Represents a multi-dimensional array (tensor) of data. The tensor is stored
 * in a contiguous block of memory, and the shape, strides, data type, number
 * of dimensions, and data pointer are stored in the struct.
 *
 * Also includes requires_grad, grad, and a pointer to the autograd graph node.
 */
typedef struct {
    mlc_storage* data;
    int64_t offset;
    int64_t shape[MLC_MAX_DIMS];
    int64_t strides[MLC_MAX_DIMS];
    mlc_dtype dtype;
    int64_t ndim;
    bool requires_grad;
    void* grad;
    void* autograd_node;
} mlc_tensor;

mlc_tensor* mlc_empty(const int64_t* shape, int64_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_zeros(const int64_t* shape, int64_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_ones(const int64_t* shape, int64_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_full(const int64_t* shape, int64_t ndim, mlc_dtype dtype,
                     float value);
mlc_tensor* mlc_from_data(const void* src, const int64_t* shape, int64_t ndim,
                          mlc_dtype dtype);
mlc_tensor* mlc_arange(double start, double end, double step, mlc_dtype dtype);
mlc_tensor* mlc_rand(mlc_rng* rng, const int64_t* shape, int64_t ndim, float min, float max);
mlc_tensor* mlc_randn(mlc_rng* rng, const int64_t* shape, int64_t ndim, float mu, float sigma);

void mlc_tensor_free(mlc_tensor* tensor);
int64_t mlc_tensor_numel(const mlc_tensor* tensor);
bool mlc_is_contiguous(const mlc_tensor* tensor);
void* mlc_tensor_data(const mlc_tensor* tensor);

mlc_tensor* mlc_view(mlc_tensor* tensor, const int64_t* shape, int64_t ndim);
mlc_tensor* mlc_reshape(mlc_tensor* tensor, const int64_t* shape, int64_t ndim);
mlc_tensor* mlc_permute(mlc_tensor* tensor, const int64_t* dims);
mlc_tensor* mlc_transpose(mlc_tensor* tensor, int64_t dim0, int64_t dim1);
mlc_tensor* mlc_slice(mlc_tensor* tensor, int64_t dim, int64_t start,
                      int64_t end, int64_t step);
mlc_tensor* mlc_expand(mlc_tensor* tensor, const int64_t* shape, int64_t ndim);
mlc_tensor* mlc_squeeze(mlc_tensor* tensor, int64_t dim);
mlc_tensor* mlc_unsqueeze(mlc_tensor* tensor, int64_t dim);

mlc_tensor* mlc_contiguous(mlc_tensor* tensor);
mlc_tensor* mlc_clone(mlc_tensor* tensor);

bool mlc_broadcast_shapes(const int64_t* shape1, int64_t ndim1,
                          const int64_t* shape2, int64_t ndim2,
                          int64_t* out_shape, int64_t* out_ndim);

void mlc_print(const mlc_tensor* tensor);

#endif
