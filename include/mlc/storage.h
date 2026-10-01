#ifndef MLC_STORAGE_H
#define MLC_STORAGE_H

#include <stdlib.h>

#include "device.h"
#include "error.h"

/*
 * Represents a contiguous block of memory that can be used to store data. The
 * storage can be on either CPU or CUDA (GPU), and the size of the storage is
 * specified in bytes. The reference count manages lifetime and deallocation.
 */
typedef struct {
    void* data;
    size_t size;
    size_t ref_count;
    mlc_device device;
} mlc_storage;

mlc_storage* mlc_storage_new(size_t size, mlc_device device);
void mlc_storage_retain(mlc_storage* storage);
void mlc_storage_release(mlc_storage* storage);

#endif
