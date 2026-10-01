#include "mlc/tensor.h"

#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mlc/device.h"
#include "mlc/error.h"
#include "mlc/storage.h"

/*
 * Internal function to calculate the number of elements in a tensor given its
 * shape and number of dimensions.
 */
static int64_t mlc_numel(const int64_t* shape, int64_t ndim) {
    int64_t numel = 1;
    for (int64_t i = 0; i < ndim; ++i) {
        numel *= shape[i];
    }
    return numel;
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
 * Internal function to calculate the total size in bytes of a tensor given its
 * shape, number of dimensions, and data type.
 */
static size_t mlc_nbytes(const int64_t* shape, int64_t ndim, mlc_dtype dtype) {
    int64_t numel = mlc_numel(shape, ndim);
    return int64_to_size(numel) * mlc_size(dtype);
}

/*
 * Creates a new tensor with the specified shape, number of dimensions, data
 * type, and storage. If storage is NULL, a new storage is allocated. The tensor
 * is initialized with the given shape and strides, and the offset is set to 0.
 */
static mlc_tensor* mlc_new_view(const int64_t* shape, int64_t ndim,
                                mlc_dtype dtype, int64_t offset,
                                mlc_storage* storage) {
    MLC_CHECK(ndim >= 0 && ndim <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS or is negative");
    for (int64_t i = 0; i < ndim; ++i) {
        MLC_CHECK(shape[i] >= 0, "Shape dimensions must be non-negative");
    }

    mlc_tensor* t = (mlc_tensor*)malloc(sizeof(mlc_tensor));
    if (t == NULL) {
        return NULL;
    }

    t->ndim = ndim;
    t->dtype = dtype;
    t->offset = offset;
    for (int64_t i = 0; i < ndim; ++i) {
        t->shape[i] = shape[i];
    }
    if (ndim > 0) {
        t->strides[ndim - 1] = 1;
        for (int64_t i = ndim - 2; i >= 0; --i) {
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
        }
    }
    if (storage == NULL) {
        size_t total_size = mlc_nbytes(shape, ndim, dtype);
        storage = mlc_storage_new(total_size, (mlc_device){MLC_DEVICE_CPU});
        if (storage == NULL) {
            free(t);
            return NULL;
        }
    } else {
        mlc_storage_retain(storage);
    }
    t->data = storage;

    t->requires_grad = false;
    t->grad = NULL;
    t->autograd_node = NULL;

    return t;
}

/*
 * Creates a new tensor with the specified shape, number of dimensions, and data
 * type. The tensor is allocated on the CPU by default. The data inside is
 * uninitialized.
 */
mlc_tensor* mlc_empty(const int64_t* shape, int64_t ndim, mlc_dtype dtype) {
    return mlc_new_view(shape, ndim, dtype, 0, NULL);
}

/*
 * Checks if the given dimension is valid for the specified number of
 * dimensions. If the dimension is negative, it is converted to a positive
 * index. If the dimension is out of range, an error is raised.
 */
static int64_t mlc_check_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) {
        dim += ndim;
    }
    MLC_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");
    return dim;
}

/*
 * Checks if the given dimension is valid for unsqueezing a tensor with the
 * specified number of dimensions. If the dimension is negative, it is
 * converted to a positive index. If the dimension is out of range, an error is
 * raised.
 */
static int64_t mlc_check_unsqueeze_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) {
        dim += ndim + 1;
    }
    MLC_CHECK(dim >= 0 && dim <= ndim, "Dimension out of range for unsqueeze");
    return dim;
}

/*
 * Creates a new tensor of zeroes. It has the specified shape, number of
 * dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_zeros(const int64_t* shape, int64_t ndim, mlc_dtype dtype) {
    mlc_tensor* tensor = mlc_new_view(shape, ndim, dtype, 0, NULL);
    if (tensor == NULL) {
        return NULL;
    }
    size_t total_size = mlc_nbytes(shape, ndim, dtype);
    memset(tensor->data->data, 0, total_size);
    return tensor;
}

/*
 * Creates a new tensor of ones. It has the specified shape, number of
 * dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_ones(const int64_t* shape, int64_t ndim, mlc_dtype dtype) {
    return mlc_full(shape, ndim, dtype, 1.0);
}

/*
 * Creates a new tensor filled with the specified value. It has the specified
 * shape, number of dimensions, and data type. The tensor is allocated on the
 * CPU by default.
 */
mlc_tensor* mlc_full(const int64_t* shape, int64_t ndim, mlc_dtype dtype,
                     double value) {
    mlc_tensor* tensor = mlc_new_view(shape, ndim, dtype, 0, NULL);
    if (tensor == NULL) {
        return NULL;
    }
    int64_t numel = mlc_numel(shape, ndim);
    for (int64_t i = 0; i < numel; ++i) {
        ((float*)tensor->data->data)[i] = (float)value;
    }
    return tensor;
}

/*
 * Creates a new tensor from the given data. It has the specified shape, number
 * of dimensions, and data type. The tensor is allocated on the CPU by default.
 */
mlc_tensor* mlc_from_data(const void* src, const int64_t* shape, int64_t ndim,
                          mlc_dtype dtype) {
    mlc_tensor* tensor = mlc_new_view(shape, ndim, dtype, 0, NULL);
    if (tensor == NULL) {
        return NULL;
    }
    size_t total_size = mlc_nbytes(shape, ndim, dtype);
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
    MLC_CHECK(step != 0.0, "Step cannot be zero for arange");
    MLC_CHECK((end - start) / step >= 0.0, "Invalid range for arange");

    int64_t numel = (int64_t)ceil((end - start) / step);
    if (numel < 0) {
        numel = 0;
    }

    int64_t shape[1] = {numel};
    mlc_tensor* tensor = mlc_new_view(shape, 1, dtype, 0, NULL);
    if (tensor == NULL) {
        return NULL;
    }
    float* data = (float*)tensor->data->data;
    for (int64_t i = 0; i < numel; ++i) {
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
int64_t mlc_tensor_numel(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    return mlc_numel(tensor->shape, tensor->ndim);
}

/*
 * Checks if the given tensor is contiguous in memory. It compares the strides
 * of the tensor with the expected strides for a contiguous tensor.
 */
bool mlc_is_contiguous(const mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    int64_t expected_stride = 1;
    for (int64_t i = tensor->ndim; i > 0; --i) {
        if (tensor->shape[i - 1] == 1) {
            continue;
        }
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
    return (unsigned char*)tensor->data->data +
           int64_to_size(tensor->offset) * mlc_size(tensor->dtype);
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor.
 */
mlc_tensor* mlc_view(mlc_tensor* tensor, const int64_t* shape, int64_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(mlc_is_contiguous(tensor), "Tensor must be contiguous for view");
    MLC_CHECK(mlc_numel(shape, ndim) == mlc_tensor_numel(tensor),
              "Unequal number of elements when viewing");

    mlc_tensor* t =
        mlc_new_view(shape, ndim, tensor->dtype, tensor->offset, tensor->data);
    if (t == NULL) {
        return NULL;
    }

    return t;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor. The number of elements in the new shape must be equal
 * to the number of elements in the original tensor.
 */
mlc_tensor* mlc_reshape(mlc_tensor* tensor, const int64_t* shape,
                        int64_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(mlc_numel(shape, ndim) == mlc_tensor_numel(tensor),
              "Unequal number of elements when reshaping");

    if (mlc_is_contiguous(tensor)) {
        return mlc_new_view(shape, ndim, tensor->dtype, tensor->offset,
                            tensor->data);
    } else {
        mlc_tensor* cont = mlc_contiguous(tensor);
        if (cont == NULL) {
            return NULL;
        }
        mlc_tensor* res =
            mlc_new_view(shape, ndim, tensor->dtype, cont->offset, cont->data);
        mlc_tensor_free(cont);
        return res;
    }
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different order of dimensions. The new tensor retains the storage of the
 * original tensor. The dims array specifies the new order of dimensions.
 */
mlc_tensor* mlc_permute(mlc_tensor* tensor, const int64_t* dims) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    bool seen[MLC_MAX_DIMS] = {false};

    mlc_tensor* new_tensor =
        mlc_new_view(tensor->shape, tensor->ndim, tensor->dtype, tensor->offset,
                     tensor->data);
    if (new_tensor == NULL) {
        return NULL;
    }

    for (int64_t i = 0; i < tensor->ndim; ++i) {
        int64_t d = mlc_check_dim(dims[i], tensor->ndim);
        MLC_CHECK(!seen[d], "Duplicate dimension in permute");
        seen[d] = true;
        new_tensor->shape[i] = tensor->shape[d];
        new_tensor->strides[i] = tensor->strides[d];
    }

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * two dimensions swapped. The new tensor retains the storage of the original
 * tensor.
 */
mlc_tensor* mlc_transpose(mlc_tensor* tensor, int64_t dim0, int64_t dim1) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    dim0 = mlc_check_dim(dim0, tensor->ndim);
    dim1 = mlc_check_dim(dim1, tensor->ndim);

    int64_t dims[MLC_MAX_DIMS];
    for (int64_t i = 0; i < tensor->ndim; ++i) {
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
mlc_tensor* mlc_slice(mlc_tensor* tensor, int64_t dim, int64_t start,
                      int64_t end, int64_t step) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(step > 0, "Slice step must be greater than zero");
    dim = mlc_check_dim(dim, tensor->ndim);

    int64_t dim_size = tensor->shape[dim];
    if (start < 0) start += dim_size;
    if (end < 0) end += dim_size;
    if (start < 0) start = 0;
    if (end > dim_size) end = dim_size;

    MLC_CHECK(start <= end, "Invalid start and end for slice");

    mlc_tensor* new_tensor =
        mlc_new_view(tensor->shape, tensor->ndim, tensor->dtype, tensor->offset,
                     tensor->data);
    if (new_tensor == NULL) {
        return NULL;
    }

    new_tensor->offset = tensor->offset + (start * tensor->strides[dim]);
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        new_tensor->shape[i] = tensor->shape[i];
        new_tensor->strides[i] = tensor->strides[i];
    }
    new_tensor->shape[dim] =
        (end > start) ? (end - start + step - 1) / step : 0;
    new_tensor->strides[dim] *= step;

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has a
 * different shape and number of dimensions. The new tensor retains the storage
 * of the original tensor. The new shape must be compatible with broadcasting
 * rules.
 */
mlc_tensor* mlc_expand(mlc_tensor* tensor, const int64_t* shape, int64_t ndim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    MLC_CHECK(ndim >= tensor->ndim,
              "Expanded number of dimensions cannot be less than original");

    mlc_tensor* new_tensor =
        mlc_new_view(shape, ndim, tensor->dtype, tensor->offset, tensor->data);
    if (new_tensor == NULL) {
        return NULL;
    }

    for (int64_t i = 0; i < ndim; ++i) {
        new_tensor->shape[i] = shape[i];
        if (i < ndim - tensor->ndim) {
            new_tensor->strides[i] = 0;
        } else {
            int64_t j = i - (ndim - tensor->ndim);
            if (tensor->shape[j] == 1 && shape[i] > 1) {
                new_tensor->strides[i] = 0;
            } else {
                MLC_CHECK(tensor->shape[j] == shape[i] || tensor->shape[j] == 1,
                          "Invalid shape for expand");
                new_tensor->strides[i] = tensor->strides[j];
            }
        }
    }

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * one less dimension. The new tensor retains the storage of the original
 * tensor. The specified dimension must have size 1 in order to be squeezed.
 */
mlc_tensor* mlc_squeeze(mlc_tensor* tensor, int64_t dim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    dim = mlc_check_dim(dim, tensor->ndim);
    MLC_CHECK(tensor->shape[dim] == 1,
              "Cannot squeeze dimension with size > 1");

    mlc_tensor* new_tensor =
        mlc_new_view(tensor->shape, tensor->ndim - 1, tensor->dtype,
                     tensor->offset, tensor->data);
    if (new_tensor == NULL) {
        return NULL;
    }

    for (int64_t i = 0, j = 0; i < tensor->ndim; ++i) {
        if (i != dim) {
            new_tensor->shape[j] = tensor->shape[i];
            new_tensor->strides[j] = tensor->strides[i];
            j++;
        }
    }

    return new_tensor;
}

/*
 * Creates a new tensor that shares the same data as the given tensor but has
 * one more dimension. The new tensor retains the storage of the original
 * tensor. The specified dimension will have size 1 in the new tensor.
 */
mlc_tensor* mlc_unsqueeze(mlc_tensor* tensor, int64_t dim) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    dim = mlc_check_unsqueeze_dim(dim, tensor->ndim);

    int64_t new_shape[MLC_MAX_DIMS];
    for (int64_t i = 0, j = 0; i < tensor->ndim + 1; ++i) {
        if (i == dim) {
            new_shape[i] = 1;
        } else {
            new_shape[i] = tensor->shape[j];
            j++;
        }
    }

    mlc_tensor* new_tensor =
        mlc_new_view(new_shape, tensor->ndim + 1, tensor->dtype, tensor->offset,
                     tensor->data);
    if (new_tensor == NULL) {
        return NULL;
    }

    for (int64_t i = 0, j = 0; i < new_tensor->ndim; ++i) {
        if (i == dim) {
            new_tensor->strides[i] = 0;
        } else {
            new_tensor->strides[i] = tensor->strides[j];
            j++;
        }
    }

    return new_tensor;
}

/*
 * Creates a new tensor that is a contiguous copy of the given tensor.
 */
mlc_tensor* mlc_contiguous(mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");

    mlc_tensor* new_tensor =
        mlc_new_view(tensor->shape, tensor->ndim, tensor->dtype, 0, NULL);
    if (new_tensor == NULL) {
        return NULL;
    }

    int64_t numel = mlc_tensor_numel(tensor);
    size_t elem_size = mlc_size(tensor->dtype);
    unsigned char* src_data = (unsigned char*)tensor->data->data;
    unsigned char* dst_data = (unsigned char*)new_tensor->data->data;

    for (int64_t i = 0; i < numel; ++i) {
        int64_t src_idx = tensor->offset;
        int64_t temp = i;
        for (int64_t k = tensor->ndim; k > 0; --k) {
            int64_t dim = k - 1;
            int64_t coord = temp % tensor->shape[dim];
            temp /= tensor->shape[dim];
            src_idx += coord * tensor->strides[dim];
        }
        memcpy(dst_data + int64_to_size(i) * elem_size,
               src_data + int64_to_size(src_idx) * elem_size, elem_size);
    }

    return new_tensor;
}

/*
 * Creates a new tensor that is a copy of the given tensor. The new tensor has
 * the same shape, number of dimensions, and data type as the original tensor.
 */
mlc_tensor* mlc_clone(mlc_tensor* tensor) {
    MLC_CHECK(tensor != NULL, "mlc_tensor is NULL");
    return mlc_contiguous(tensor);
}

/*
 * Broadcasts two shapes according to broadcasting rules. The resulting shape is
 * stored in out_shape and the number of dimensions in out_ndim. Returns true if
 * the shapes are broadcastable, false otherwise.
 */
bool mlc_broadcast_shapes(const int64_t* shape1, int64_t ndim1,
                          const int64_t* shape2, int64_t ndim2,
                          int64_t* out_shape, int64_t* out_ndim) {
    MLC_CHECK(ndim1 <= MLC_MAX_DIMS && ndim2 <= MLC_MAX_DIMS,
              "Number of dimensions exceeds MLC_MAX_DIMS");

    *out_ndim = (ndim1 > ndim2) ? ndim1 : ndim2;
    for (int64_t i = 0; i < *out_ndim; ++i) {
        int64_t dim1 =
            (i < *out_ndim - ndim1) ? 1 : shape1[i - (*out_ndim - ndim1)];
        int64_t dim2 =
            (i < *out_ndim - ndim2) ? 1 : shape2[i - (*out_ndim - ndim2)];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            return false;
        }
        out_shape[i] = (dim1 == 1) ? dim2 : dim1;
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
    for (int64_t i = 0; i < tensor->ndim; ++i) {
        printf("%" PRId64, tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("], dtype=%s, device=%s)\n", mlc_dtype_str(tensor->dtype),
           mlc_device_str(tensor->data->device));

    int64_t numel = mlc_tensor_numel(tensor);
    unsigned char* src_data = (unsigned char*)tensor->data->data;
    size_t elem_size = mlc_size(tensor->dtype);

    printf("Values: [");
    for (int64_t i = 0; i < numel; ++i) {
        int64_t src_idx = tensor->offset;
        int64_t temp = i;
        for (int64_t k = tensor->ndim; k > 0; --k) {
            int64_t dim = k - 1;
            int64_t coord =
                (tensor->shape[dim] > 0) ? (temp % tensor->shape[dim]) : 0;
            if (tensor->shape[dim] > 0) temp /= tensor->shape[dim];
            src_idx += coord * tensor->strides[dim];
        }
        float val = *((float*)(src_data + int64_to_size(src_idx) * elem_size));
        printf("%f", val);
        if (i < numel - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}
