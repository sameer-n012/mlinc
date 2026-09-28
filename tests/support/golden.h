/*
 * golden.h - reader for golden test data files (.mlct).
 *
 * The Python reference project (tests/ref/mlinc_ref/golden.py) writes these
 * files. The C tests read them to get inputs and expected outputs.
 *
 * File format, version 1. All integers are little-endian.
 *
 *   file   := header entry{count}
 *   header := u8  magic[4] = "MLCT"
 *             u32 version  = 1
 *             u32 count                   number of entries
 *   entry  := u32 name_len                1 .. GOLDEN_MAX_NAME
 *             u8  name[name_len]          UTF-8, no NUL, unique in the file
 *             u32 dtype                   golden_dtype value
 *             u32 ndim                    0 .. GOLDEN_MAX_DIMS (0 = scalar)
 *             i64 shape[ndim]             each dim >= 0
 *             u8  data[numel * size]      row-major, contiguous, no padding
 *
 * The file must end directly after the last entry.
 */
#ifndef MLC_TESTS_GOLDEN_H
#define MLC_TESTS_GOLDEN_H

#include <stddef.h>
#include <stdint.h>

/* The Makefile can override this with -DMLC_TEST_DATA_DIR='"..."'.
 * The default is correct when the tests run from the repository root. */
#ifndef MLC_TEST_DATA_DIR
#define MLC_TEST_DATA_DIR "tests/data"
#endif

#define GOLDEN_MAX_NAME 63
#define GOLDEN_MAX_DIMS 8

typedef enum {
    GOLDEN_F32 = 0,
    GOLDEN_F64 = 1,
    GOLDEN_I32 = 2,
    GOLDEN_I64 = 3,
} golden_dtype;

typedef struct {
    char name[GOLDEN_MAX_NAME + 1]; /* NUL-terminated */
    golden_dtype dtype;
    uint32_t ndim;
    int64_t shape[GOLDEN_MAX_DIMS];
    int64_t numel; /* product of shape; 1 for a scalar */
    void* data;    /* 64-byte aligned; the golden_file owns it */
} golden_tensor;

typedef struct {
    uint32_t count;
    golden_tensor* tensors; /* the golden_file owns it */
} golden_file;

/* Returns the size in bytes of one element, or 0 for an unknown dtype. */
size_t golden_dtype_size(golden_dtype dtype);

/* Returns the name of the dtype ("f32", ...), or "unknown". */
const char* golden_dtype_name(golden_dtype dtype);

/*
 * Loads the file at `path` into `out`.
 * Returns 0 on success. On failure, returns -1, writes a message to `err` (if
 * `err` is not NULL), and leaves `out` empty (no memory to free). On success,
 * the caller must call golden_free(out).
 */
int golden_load(const char* path, golden_file* out, char* err, size_t err_len);

/* Returns the entry with this name, or NULL. The golden_file keeps ownership.
 */
const golden_tensor* golden_get(const golden_file* file, const char* name);

/*
 * Like golden_get, but for tests: prints a message and aborts if the entry does
 * not exist or if its dtype is not `dtype`.
 */
const golden_tensor* golden_require(const golden_file* file, const char* name,
                                    golden_dtype dtype);

/* Frees all memory of `file` and sets it to empty. Safe to call on an empty
 * golden_file. */
void golden_free(golden_file* file);

#endif /* MLC_TESTS_GOLDEN_H */
