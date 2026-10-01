#ifndef MLC_DTYPE_H
#define MLC_DTYPE_H

#include <stdlib.h>

typedef enum {
    MLC_F32,
} mlc_dtype;

/*
 * Returns the size in bytes of the given data type.
 */
static inline size_t sizeof_dtype(mlc_dtype dtype) {
    switch (dtype) {
        case MLC_F32:
            return sizeof(float);
    }
    return 0;
}

/*
 * Returns the string representation of the given data type.
 */
static inline const char* mlc_dtype_str(mlc_dtype dtype) {
    switch (dtype) {
        case MLC_F32:
            return "float32";
    }
    return "unknown";
}

#endif
