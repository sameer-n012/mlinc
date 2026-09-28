#ifndef MLC_ERROR_H
#define MLC_ERROR_H

#include <stdio.h>
#include <stdlib.h>

#define MLC_CHECK(cond, ...)                                            \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "[MLC CHECK] %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                               \
            fprintf(stderr, "\n");                                      \
            abort();                                                    \
        }                                                               \
    } while (0)

typedef enum {
    MLC_SUCCESS = 0,
    MLC_ERROR_INVALID_ARGUMENT,
    MLC_ERROR_OUT_OF_MEMORY,
    MLC_ERROR_FILE_IO,
    MLC_ERROR_CUDA_RUNTIME,
    MLC_ERROR_UNKNOWN,
} mlc_status;

static inline const char* mlc_status_str(mlc_status status) {
    switch (status) {
        case MLC_SUCCESS:
            return "Success";
        case MLC_ERROR_INVALID_ARGUMENT:
            return "Invalid argument";
        case MLC_ERROR_OUT_OF_MEMORY:
            return "Out of memory";
        case MLC_ERROR_FILE_IO:
            return "File I/O error";
        case MLC_ERROR_CUDA_RUNTIME:
            return "CUDA runtime error";
        case MLC_ERROR_UNKNOWN:
            return "Unknown error";
    }
    return "Unrecognized error code";
}

#endif
