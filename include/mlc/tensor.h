#ifndef MLC_TENSOR_H
#define MLC_TENSOR_H

#include <stdbool.h>
#include <stdlib.h>

#include "dtype.h"
#include "error.h"
#include "storage.h"

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
    size_t offset;
    size_t shape[MLC_MAX_DIMS];
    size_t strides[MLC_MAX_DIMS];
    mlc_dtype dtype;
    size_t ndim;
    bool requires_grad;
    void* grad;
    void* autograd_node;
} mlc_tensor;

mlc_tensor* mlc_empty(size_t* shape, size_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_zeros(size_t* shape, size_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_ones(size_t* shape, size_t ndim, mlc_dtype dtype);
mlc_tensor* mlc_full(size_t* shape, size_t ndim, mlc_dtype dtype, void* value);
mlc_tensor* mlc_from_data(const void* src, size_t* shape, size_t ndim,
                          mlc_dtype dtype);
mlc_tensor* mlc_arange(double start, double end, double step, mlc_dtype dtype);

void mlc_tensor_free(mlc_tensor* tensor);
size_t mlc_tensor_numel(const mlc_tensor* tensor);
bool mlc_is_contiguous(const mlc_tensor* tensor);
void* mlc_tensor_data(const mlc_tensor* tensor);

mlc_tensor* mlc_view(mlc_tensor* tensor, size_t* shape, size_t ndim);
mlc_tensor* mlc_reshape(mlc_tensor* tensor, size_t* shape, size_t ndim);
mlc_tensor* mlc_permute(mlc_tensor* tensor, const size_t* dims);
mlc_tensor* mlc_transpose(mlc_tensor* tensor, size_t dim0, size_t dim1);
mlc_tensor* mlc_slice(mlc_tensor* tensor, size_t dim, size_t start, size_t end,
                      size_t step);
mlc_tensor* mlc_expand(mlc_tensor* tensor, size_t* shape, size_t ndim);
mlc_tensor* mlc_squeeze(mlc_tensor* tensor, size_t dim);
mlc_tensor* mlc_unsqueeze(mlc_tensor* tensor, size_t dim);

mlc_tensor* mlc_contiguous(mlc_tensor* tensor);
mlc_tensor* mlc_clone(mlc_tensor* tensor);

bool mlc_broadcast_shapes(const size_t* shape1, size_t ndim1,
                          const size_t* shape2, size_t ndim2, size_t* out_shape,
                          size_t* out_ndim);

void mlc_print(const mlc_tensor* tensor);

#endif
