#include "alloc.h"

#include <stdlib.h>

#include "device.h"
#include "error.h"

/*
 * Allocates memory on the specified device. For CPU, it uses aligned_alloc to
 * allocate memory aligned to 64 bytes. The size is rounded up to the nearest
 * multiple of 64 bytes, and at least one block is allocated to ensure a valid
 * non-NULL pointer.
 */
void* mlc_alloc(size_t size, mlc_device device) {
    size_t alloc_size =
        (size + MLC_ALIGN_BLOCK - 1) / MLC_ALIGN_BLOCK * MLC_ALIGN_BLOCK;
    if (alloc_size == 0) {
        alloc_size = MLC_ALIGN_BLOCK;
    }

    if (device == MLC_DEVICE_CPU) {
        return aligned_alloc(MLC_ALIGN_BLOCK, alloc_size);
    }

    return NULL;
}

/*
 * Frees memory on the specified device. For CPU, it uses free to deallocate
 * the memory.
 */
void mlc_free(void* ptr, mlc_device device) {
    if (ptr == NULL) {
        return;
    }

    if (device == MLC_DEVICE_CPU) {
        free(ptr);
    }
}
