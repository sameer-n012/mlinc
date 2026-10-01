#ifndef MLC_ERROR_H
#define MLC_ERROR_H

#include <stdio.h>
#include <stdlib.h>

/*
 * Checks a condition and aborts the program with an error message if the
 * condition is false. The error message is printed to stderr and includes the
 * file name and line number where the check failed.
 */
#define MLC_CHECK(cond, ...)                                            \
    do {                                                                \
        if (!(cond)) {                                                  \
            fprintf(stderr, "[MLC CHECK] %s:%d: ", __FILE__, __LINE__); \
            fprintf(stderr, __VA_ARGS__);                               \
            fprintf(stderr, "\n");                                      \
            abort();                                                    \
        }                                                               \
    } while (0)

/*
 * Error codes returned by MLC functions. Functions that can fail return an
 * mlc_status value, which is MLC_SUCCESS on success and a non-zero error code
 * on failure.
 */
typedef enum {
    MLC_SUCCESS = 0,
    MLC_ERROR_INVALID_ARGUMENT,
    MLC_ERROR_OUT_OF_MEMORY,
    MLC_ERROR_FILE_IO,
    MLC_ERROR_CUDA_RUNTIME,
    MLC_ERROR_UNKNOWN,
} mlc_status;

/*
 * Returns a string representation of the given mlc_status code.
 */
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
