#ifndef MLC_ALLOC_H
#define MLC_ALLOC_H

#include "mlc/device.h"
#include "mlc/error.h"
#include <stdlib.h>

#define MLC_ALIGN_BLOCK 64

void* mlc_alloc(size_t size, mlc_device device);
void mlc_free(void* ptr, mlc_device device);

#endif
