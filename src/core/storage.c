#include "storage.h"

#include <stdlib.h>

#include "alloc.h"
#include "device.h"
#include "error.h"

/*
 * Creates a new storage with the specified size in bytes and device. The
 * storage is allocated on the specified device (CPU or CUDA). The reference
 * count is initialized to 1. If the allocation fails, NULL is returned.
 */
mlc_storage* mlc_storage_new(size_t size, mlc_device device) {
    mlc_storage* mlcs = (mlc_storage*)malloc(sizeof(mlc_storage));
    if (mlcs == NULL) {
        return NULL;
    }

    mlcs->data = mlc_alloc(size, device);
    if (mlcs->data == NULL) {
        free(mlcs);
        return NULL;
    }

    mlcs->size = size;
    mlcs->ref_count = 1;
    mlcs->device = device;

    return mlcs;
}

/*
 * Increments the reference count of the given storage. This function should be
 * called when a new reference to the storage is created, ensuring that the
 * storage is not deallocated while still in use.
 */
void mlc_storage_retain(mlc_storage* storage) {
    MLC_CHECK(storage != NULL, "mlc_storage is NULL");
    storage->ref_count++;
}

/*
 * Decrements the reference count of the given storage. If the reference count
 * reaches zero, the storage is deallocated and its memory is freed. This
 * function should be called when a reference to the storage is no longer
 * needed.
 */
void mlc_storage_release(mlc_storage* storage) {
    if (storage == NULL) {
        return;
    }

    if (storage->ref_count <= 0) {
        MLC_ERROR("mlc_storage reference count is already zero");
    }

    storage->ref_count--;
    if (storage->ref_count == 0) {
        mlc_free(storage->data, storage->device);
        free(storage);
    }
}
