#ifndef MLC_DTYPE_H
#define MLC_DTYPE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    MLC_F32,
    MLC_I64
} mlc_dtype;

/*
 * Returns the size in bytes of the given data type.
 */
static inline size_t mlc_size(mlc_dtype dtype) {
    switch (dtype) {
        case MLC_F32:
            return sizeof(float);
        case MLC_I64:
            return sizeof(int64_t);
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
        case MLC_I64:
            return "int64";
    }
    return "unknown";
}

#endif
