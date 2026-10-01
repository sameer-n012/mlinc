#include "tensor.h"

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "alloc.h"
#include "device.h"
#include "error.h"
#include "storage.h"

/*
 * Internal function to calculate the number of elements in a tensor given its
 * shape and number of dimensions.
 */
static size_t mlc_numel(const size_t* shape, size_t ndim) {
    size_t numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }
    return numel;
}

/*
 * Internal function to calculate the total size in bytes of a tensor given its
 * shape, number of dimensions, and data type.
 */
static size_t mlc_total_size(const size_t* shape, size_t ndim,
                             mlc_dtype dtype) {
    return mlc_numel(shape, ndim) * sizeof_dtype(dtype);
}

/*
 * Creates a new tensor with the specified shape, number of dimensions, and data
 * type. The tensor is allocated on the CPU by default. The data inside is
 * uninitialized.
 */
mlc_tensor* mlc_empty(size_t* shape, size_t ndim, mlc_dtype dtype) {
    MLC_CHECK(ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    mlc_tensor* tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(tensor != NULL, "Failed to allocate memory for mlc_tensor");

    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->offset = 0;
    for (size_t i = 0; i < ndim; ++i) {
        tensor->shape[i] = shape[i];
    }
    tensor->strides[ndim - 1] = 1;
    for (size_t i = ndim - 2; i >= 0; --i) {
        tensor->strides[i] = tensor->strides[i + 1] * tensor->shape[i + 1];
    }

    size_t total_size = mlc_total_size(shape, ndim, dtype);

    mlc_storage* storage =
        mlc_storage_new(total_size, (mlc_device){MLC_DEVICE_CPU});
    tensor->data = storage;

    tensor->requires_grad = false;
    tensor->grad = NULL;
    tensor->autograd_node = NULL;

    return tensor;
}

/*
 * Creates a new tensor of zeroes. It has the specified shape, number of
 * dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_zeros(size_t* shape, size_t ndim, mlc_dtype dtype) {
    mlc_tensor* tensor = mlc_empty(shape, ndim, dtype);
    size_t total_size = mlc_total_size(shape, ndim, dtype);
    memset(tensor->data->data, 0.0f, total_size);
    return tensor;
}

/*
 * Creates a new tensor of ones. It has the specified shape, number of
 * dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_ones(size_t* shape, size_t ndim, mlc_dtype dtype) {
    mlc_tensor* tensor = mlc_empty(shape, ndim, dtype);
    size_t total_size = mlc_total_size(shape, ndim, dtype);
    for (size_t i = 0; i < total_size / sizeof_dtype(dtype); ++i) {
        ((float*)tensor->data->data)[i] = 1.0f;
    }
    return tensor;
}

/*
 * Creates a new tensor filled with the specified value. It has the specified
 * shape, number of dimensions, and data type. The tensor is allocated on the
 * CPU by default.
 */
mlc_tensor* mlc_full(size_t* shape, size_t ndim, mlc_dtype dtype,
                     double value) {
    mlc_tensor* tensor = mlc_empty(shape, ndim, dtype);
    size_t total_size = mlc_total_size(shape, ndim, dtype);
    for (size_t i = 0; i < total_size / sizeof_dtype(dtype); ++i) {
        ((float*)tensor->data->data)[i] = (float)value;
    }
    return tensor;
}

/*
 * Creates a new tensor from the given data. It has the specified shape, number
 * of dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_from_data(const void* src, size_t* shape, size_t ndim,
                          mlc_dtype dtype) {
    mlc_tensor* tensor = mlc_empty(shape, ndim, dtype);
    size_t total_size = mlc_total_size(shape, ndim, dtype);
    memcpy(tensor->data->data, src, total_size);
    return tensor;
}

/*
 * Creates a new tensor with values in the range [start, end) with the
 * specified step size. It has the specified data type. The tensor is allocated
 * on the CPU by default. The number of elements is calculated as
 * (end - start) / step.
 */
mlc_tensor* mlc_arange(double start, double end, double step, mlc_dtype dtype) {
    size_t numel = (size_t)((end - start) / step);
    size_t shape[1] = {numel};
    mlc_tensor* tensor = mlc_empty(shape, 1, dtype);
    float* data = (float*)tensor->data->data;
    for (size_t i = 0; i < numel; ++i) {
        data[i] = (float)(start + (double)i * step);
    }
    return tensor;
}

/*
 * Frees the memory associated with the given tensor. It dereferences the
 * storage and frees the tensor struct itself.
 */
void mlc_tensor_free(mlc_tensor* tensor) {
    if (tensor == NULL) {
        return;
    }
    mlc_storage_release(tensor->data);
    free(tensor);
}

/*
 * Returns the total number of elements in the given tensor. It calculates the
 * product of the shape dimensions.
 */
size_t mlc_tensor_numel(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    return mlc_numel(tensor->shape, tensor->ndim);
}

/*
 * Checks if the given tensor is contiguous in memory. It compares the strides
 * of the tensor with the expected strides for a contiguous tensor.
 */
bool mlc_is_contiguous(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    size_t expected_stride = 1;
    for (size_t i = tensor->ndim; i > 0; --i) {
        if (tensor->strides[i - 1] != expected_stride) {
            return false;
        }
        expected_stride *= tensor->shape[i - 1];
    }
    return true;
}

/*
 * Returns a pointer to the data of the given tensor.
 */
void* mlc_tensor_data(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    return tensor->data->data + tensor->offset * sizeof_dtype(tensor->dtype);
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor.
 */
mlc_tensor* mlc_view(mlc_tensor* tensor, size_t* shape, size_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");
    MLC_CHECK(mlc_numel(shape, ndim) == mlc_tensor_numel(tensor),
              "Unequal number of elements when viewing");

    mlc_tensor* t = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    t->data = tensor->data;
    mlc_storage_retain(t->data);
    t->ndim = ndim;
    t->dtype = tensor->dtype;
    t->offset = tensor->offset;
    for (size_t i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
        t->strides[i] = tensor->strides[i];
    }

    t->requires_grad = tensor->requires_grad;
    t->grad = tensor->grad;
    t->autograd_node = tensor->autograd_node;

    return t;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor. The number of elements in the new shape must be equal
 * to the number of elements in the original tensor.
 */
mlc_tensor* mlc_reshape(mlc_tensor* tensor, size_t* shape, size_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    size_t new_numel = mlc_numel(shape, ndim);
    size_t old_numel = mlc_tensor_numel(tensor);
    MLC_CHECK(new_numel == old_numel,
              "Unequal number of elements when reshaping");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = ndim;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset;
    for (size_t i = 0; i < ndim; ++i) {
        new_tensor->shape[i] = shape[i];
        new_tensor->strides[i] = tensor->strides[i];
    }

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different order of dimensions. The new tensor retains the storage of the
 * original tensor. The dims array specifies the new order of dimensions.
 */
mlc_tensor* mlc_permute(mlc_tensor* tensor, const size_t* dims) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(tensor->ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = tensor->ndim;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset;
    for (size_t i = 0; i < tensor->ndim; ++i) {
        new_tensor->shape[i] = tensor->shape[dims[i]];
        new_tensor->strides[i] = tensor->strides[dims[i]];
    }

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * two dimensions swapped. The new tensor retains the storage of the original
 * tensor.
 */
mlc_tensor* mlc_transpose(mlc_tensor* tensor, size_t dim0, size_t dim1) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(dim0 < tensor->ndim && dim1 < tensor->ndim,
              "Invalid dimensions for transpose");

    size_t dims[MLC_MAX_DIMS];
    for (size_t i = 0; i < tensor->ndim; ++i) {
        dims[i] = i;
    }
    dims[dim0] = dim1;
    dims[dim1] = dim0;

    return mlc_permute(tensor, dims);
}

/*
 * Creates a new tensor that shares the same data as the given tensor but is a
 * slice of the original tensor along the specified dimension. The new tensor
 * retains the storage of the original tensor. The start, end, and step
 * parameters specify the range of indices to include in the slice.
 */
mlc_tensor* mlc_slice(mlc_tensor* tensor, size_t dim, size_t start, size_t end,
                      size_t step) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(dim < tensor->ndim, "Invalid dimension for slice");
    MLC_CHECK(start < end && end <= tensor->shape[dim],
              "Invalid start and end for slice");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = tensor->ndim;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset + start * tensor->strides[dim];
    for (size_t i = 0; i < tensor->ndim; ++i) {
        new_tensor->shape[i] = tensor->shape[i];
        new_tensor->strides[i] = tensor->strides[i];
    }
    new_tensor->shape[dim] = (end - start + step - 1) / step;
    new_tensor->strides[dim] *= step;

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor. The new shape must be compatible with broadcasting
 * rules.
 */
mlc_tensor* mlc_expand(mlc_tensor* tensor, size_t* shape, size_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = ndim;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset;
    for (size_t i = 0; i < ndim; ++i) {
        if (i < tensor->ndim) {
            new_tensor->shape[i] = shape[i];
            new_tensor->strides[i] = tensor->strides[i];
        } else {
            new_tensor->shape[i] = shape[i];
            new_tensor->strides[i] = 0;
        }
    }

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * one less dimension. The new tensor retains the storage of the original
 * tensor. The specified dimension must have size 1 in order to be squeezed.
 */
mlc_tensor* mlc_squeeze(mlc_tensor* tensor, size_t dim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(dim < tensor->ndim, "Invalid dimension for squeeze");
    MLC_CHECK(tensor->shape[dim] == 1,
              "Cannot squeeze dimension with size > 1");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = tensor->ndim - 1;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset;
    for (size_t i = 0, j = 0; i < tensor->ndim; ++i) {
        if (i != dim) {
            new_tensor->shape[j] = tensor->shape[i];
            new_tensor->strides[j] = tensor->strides[i];
            j++;
        }
    }

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * one more dimension. The new tensor retains the storage of the original
 * tensor. The specified dimension will have size 1 in the new tensor.
 */
mlc_tensor* mlc_unsqueeze(mlc_tensor* tensor, size_t dim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(dim <= tensor->ndim, "Invalid dimension for unsqueeze");

    mlc_tensor* new_tensor = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    MLC_CHECK(new_tensor != NULL, "Failed to allocate memory for mlc_tensor");

    new_tensor->data = tensor->data;
    mlc_storage_retain(new_tensor->data);

    new_tensor->ndim = tensor->ndim + 1;
    new_tensor->dtype = tensor->dtype;
    new_tensor->offset = tensor->offset;
    for (size_t i = 0, j = 0; i < new_tensor->ndim; ++i) {
        if (i == dim) {
            new_tensor->shape[i] = 1;
            new_tensor->strides[i] = 0;
        } else {
            new_tensor->shape[i] = tensor->shape[j];
            new_tensor->strides[i] = tensor->strides[j];
            j++;
        }
    }

    new_tensor->requires_grad = tensor->requires_grad;
    new_tensor->grad = tensor->grad;
    new_tensor->autograd_node = tensor->autograd_node;

    return new_tensor;
}

/*
 * Creates a new tensor that is a contiguous copy of the given tensor.
 */
mlc_tensor* mlc_contiguous(mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");

    mlc_tensor* new_tensor =
        mlc_empty(tensor->shape, tensor->ndim, tensor->dtype);
    size_t total_size =
        mlc_total_size(tensor->shape, tensor->ndim, tensor->dtype);
    memcpy(new_tensor->data->data, tensor->data->data, total_size);

    return new_tensor;
}

/*
 * Creates a new tensor that is a copy of the given tensor. The new tensor has
 * the same shape, number of dimensions, and data type as the original tensor.
 */
mlc_tensor* mlc_clone(mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");

    mlc_tensor* new_tensor =
        mlc_empty(tensor->shape, tensor->ndim, tensor->dtype);
    size_t total_size =
        mlc_total_size(tensor->shape, tensor->ndim, tensor->dtype);
    memcpy(new_tensor->data->data, tensor->data->data, total_size);

    return new_tensor;
}

/*
 * Broadcasts two shapes according to broadcasting rules. The resulting shape is
 * stored in out_shape and the number of dimensions in out_ndim. Returns true if
 * the shapes are broadcastable, false otherwise.
 */
bool mlc_broadcast_shapes(const size_t* shape1, size_t ndim1,
                          const size_t* shape2, size_t ndim2, size_t* out_shape,
                          size_t* out_ndim) {
    MLC_CHECK(ndim1 <= MLC_MAX_DIMS && ndim2 <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    *out_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    for (size_t i = 0; i < *out_ndim; ++i) {
        size_t dim1 =
            (i < *out_ndim - ndim1) ? 1 : shape1[i - (*out_ndim - ndim1)];
        size_t dim2 =
            (i < *out_ndim - ndim2) ? 1 : shape2[i - (*out_ndim - ndim2)];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
        out_shape[i] = (dim1 > dim2) ? dim1 : dim2;
    }
    return true;
}

/*
 * Prints the details of the given tensor, including its shape, data type,
 * values, and device.
 */
void mlc_print(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");

    printf("Tensor(shape=[");
    for (size_t i = 0; i < tensor->ndim; ++i) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("], dtype=%s, device=%s)\n", mlc_dtype_str(tensor->dtype),
           mlc_device_str(tensor->data->device));

    size_t numel = mlc_tensor_numel(tensor);
    float* data = (float*)tensor->data->data;
    printf("Values: [");
    for (size_t i = 0; i < numel; ++i) {
        printf("%f", data[i]);
        if (i < numel - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}
