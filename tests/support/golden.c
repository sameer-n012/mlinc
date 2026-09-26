/*
 * golden.c - reader for golden test data files (.mlct). See golden.h for the format.
 *
 * The reader validates all header fields before it allocates memory, so a corrupt or
 * truncated file gives an error message and not a crash or a huge allocation.
 */
#include "golden.h"

#include <errno.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GOLDEN_VERSION 1u
#define GOLDEN_ALIGN ((size_t)64)
/* A sanity limit. A real golden file has a few entries, not thousands. */
#define GOLDEN_MAX_ENTRIES 4096u

static const char GOLDEN_MAGIC[4] = {'M', 'L', 'C', 'T'};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 3, 4)))
#endif
static void set_err(char *err, size_t err_len, const char *fmt, ...) {
    if (err == NULL || err_len == 0) {
        return;
    }
    va_list args;
    va_start(args, fmt);
    vsnprintf(err, err_len, fmt, args);
    va_end(args);
}

/* The reader copies integers directly from the file, so the host must be little-endian.
 * This is true for arm64 macOS and x86_64 Linux, the two target platforms. */
static int host_is_little_endian(void) {
    const uint16_t probe = 1;
    unsigned char first;
    memcpy(&first, &probe, 1);
    return first == 1;
}

static int read_exact(FILE *fp, void *buf, size_t n) {
    return fread(buf, 1, n, fp) == n ? 0 : -1;
}

static int read_u32(FILE *fp, uint32_t *value) {
    return read_exact(fp, value, sizeof *value);
}

static int read_i64(FILE *fp, int64_t *value) {
    return read_exact(fp, value, sizeof *value);
}

size_t golden_dtype_size(golden_dtype dtype) {
    switch (dtype) {
    case GOLDEN_F32:
    case GOLDEN_I32:
        return 4;
    case GOLDEN_F64:
    case GOLDEN_I64:
        return 8;
    }
    return 0;
}

const char *golden_dtype_name(golden_dtype dtype) {
    switch (dtype) {
    case GOLDEN_F32: return "f32";
    case GOLDEN_F64: return "f64";
    case GOLDEN_I32: return "i32";
    case GOLDEN_I64: return "i64";
    }
    return "unknown";
}

/* Reads one entry into `t`. The function uses `file_size` to reject a data size that is
 * larger than the unread part of the file, before it allocates memory. */
static int read_entry(FILE *fp, long file_size, golden_tensor *t, const char *path,
                      uint32_t index, char *err, size_t err_len) {
    uint32_t name_len;
    if (read_u32(fp, &name_len) != 0) {
        goto truncated;
    }
    if (name_len == 0 || name_len > GOLDEN_MAX_NAME) {
        set_err(err, err_len, "%s: entry %" PRIu32 ": name length %" PRIu32
                " is not in range [1, %d]", path, index, name_len, GOLDEN_MAX_NAME);
        return -1;
    }
    if (read_exact(fp, t->name, name_len) != 0) {
        goto truncated;
    }
    t->name[name_len] = '\0';
    if (memchr(t->name, '\0', name_len) != NULL) {
        set_err(err, err_len, "%s: entry %" PRIu32 ": name contains a NUL byte", path, index);
        return -1;
    }

    uint32_t dtype;
    if (read_u32(fp, &dtype) != 0) {
        goto truncated;
    }
    if (dtype > (uint32_t)GOLDEN_I64) {
        set_err(err, err_len, "%s: entry '%s': unknown dtype %" PRIu32, path, t->name, dtype);
        return -1;
    }
    t->dtype = (golden_dtype)dtype;

    uint32_t ndim;
    if (read_u32(fp, &ndim) != 0) {
        goto truncated;
    }
    if (ndim > GOLDEN_MAX_DIMS) {
        set_err(err, err_len, "%s: entry '%s': ndim %" PRIu32 " is larger than %d",
                path, t->name, ndim, GOLDEN_MAX_DIMS);
        return -1;
    }
    t->ndim = ndim;

    int64_t numel = 1;
    for (uint32_t d = 0; d < ndim; d++) {
        int64_t dim;
        if (read_i64(fp, &dim) != 0) {
            goto truncated;
        }
        if (dim < 0) {
            set_err(err, err_len, "%s: entry '%s': dim %" PRIu32 " is negative (%" PRId64 ")",
                    path, t->name, d, dim);
            return -1;
        }
        if (dim != 0 && numel > INT64_MAX / dim) {
            set_err(err, err_len, "%s: entry '%s': element count overflows int64",
                    path, t->name);
            return -1;
        }
        t->shape[d] = dim;
        numel *= dim;
    }
    t->numel = numel;

    /* Compare the data size with the unread part of the file before the allocation. */
    const size_t elem_size = golden_dtype_size(t->dtype);
    const long pos = ftell(fp);
    if (pos < 0) {
        set_err(err, err_len, "%s: ftell failed: %s", path, strerror(errno));
        return -1;
    }
    const uint64_t remaining = (uint64_t)(file_size - pos);
    if ((uint64_t)numel > remaining / elem_size) {
        goto truncated;
    }
    const size_t nbytes = (size_t)numel * elem_size;

    /* aligned_alloc needs a size that is a multiple of the alignment. Allocate at least one
     * block, so that an empty tensor also has a valid non-NULL pointer. */
    size_t alloc_size = (nbytes + GOLDEN_ALIGN - 1) / GOLDEN_ALIGN * GOLDEN_ALIGN;
    if (alloc_size == 0) {
        alloc_size = GOLDEN_ALIGN;
    }
    t->data = aligned_alloc(GOLDEN_ALIGN, alloc_size);
    if (t->data == NULL) {
        set_err(err, err_len, "%s: entry '%s': out of memory (%zu bytes)",
                path, t->name, alloc_size);
        return -1;
    }
    if (nbytes > 0 && read_exact(fp, t->data, nbytes) != 0) {
        goto truncated;
    }
    return 0;

truncated:
    set_err(err, err_len, "%s: entry %" PRIu32 ": file is truncated", path, index);
    return -1;
}

int golden_load(const char *path, golden_file *out, char *err, size_t err_len) {
    if (path == NULL || out == NULL) {
        set_err(err, err_len, "golden_load: path and out must not be NULL");
        return -1;
    }
    memset(out, 0, sizeof *out);
    if (!host_is_little_endian()) {
        set_err(err, err_len, "%s: big-endian hosts are not supported", path);
        return -1;
    }

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        set_err(err, err_len, "%s: cannot open: %s", path, strerror(errno));
        return -1;
    }

    long file_size = -1;
    if (fseek(fp, 0, SEEK_END) == 0) {
        file_size = ftell(fp);
    }
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        set_err(err, err_len, "%s: cannot get the file size: %s", path, strerror(errno));
        goto fail;
    }

    char magic[sizeof GOLDEN_MAGIC];
    if (read_exact(fp, magic, sizeof magic) != 0 ||
        memcmp(magic, GOLDEN_MAGIC, sizeof magic) != 0) {
        set_err(err, err_len, "%s: not a golden file (bad magic)", path);
        goto fail;
    }
    uint32_t version;
    if (read_u32(fp, &version) != 0) {
        set_err(err, err_len, "%s: file is truncated in the header", path);
        goto fail;
    }
    if (version != GOLDEN_VERSION) {
        set_err(err, err_len, "%s: version %" PRIu32 " is not supported (expected %u)",
                path, version, GOLDEN_VERSION);
        goto fail;
    }
    uint32_t count;
    if (read_u32(fp, &count) != 0) {
        set_err(err, err_len, "%s: file is truncated in the header", path);
        goto fail;
    }
    if (count > GOLDEN_MAX_ENTRIES) {
        set_err(err, err_len, "%s: entry count %" PRIu32 " is larger than %u",
                path, count, GOLDEN_MAX_ENTRIES);
        goto fail;
    }

    /* calloc sets every data pointer to NULL, so golden_free is safe after a partial load. */
    out->tensors = calloc(count > 0 ? count : 1, sizeof *out->tensors);
    if (out->tensors == NULL) {
        set_err(err, err_len, "%s: out of memory", path);
        goto fail;
    }
    out->count = count;

    for (uint32_t i = 0; i < count; i++) {
        if (read_entry(fp, file_size, &out->tensors[i], path, i, err, err_len) != 0) {
            goto fail;
        }
        for (uint32_t j = 0; j < i; j++) {
            if (strcmp(out->tensors[j].name, out->tensors[i].name) == 0) {
                set_err(err, err_len, "%s: duplicate entry name '%s'",
                        path, out->tensors[i].name);
                goto fail;
            }
        }
    }

    if (ftell(fp) != file_size) {
        set_err(err, err_len, "%s: unexpected bytes after the last entry", path);
        goto fail;
    }

    fclose(fp);
    return 0;

fail:
    fclose(fp);
    golden_free(out);
    return -1;
}

const golden_tensor *golden_get(const golden_file *file, const char *name) {
    if (file == NULL || name == NULL) {
        return NULL;
    }
    for (uint32_t i = 0; i < file->count; i++) {
        if (strcmp(file->tensors[i].name, name) == 0) {
            return &file->tensors[i];
        }
    }
    return NULL;
}

const golden_tensor *golden_require(const golden_file *file, const char *name,
                                    golden_dtype dtype) {
    const golden_tensor *t = golden_get(file, name);
    if (t == NULL) {
        fprintf(stderr, "golden_require: no entry '%s'\n", name != NULL ? name : "(null)");
        abort();
    }
    if (t->dtype != dtype) {
        fprintf(stderr, "golden_require: entry '%s' has dtype %s, expected %s\n",
                name, golden_dtype_name(t->dtype), golden_dtype_name(dtype));
        abort();
    }
    return t;
}

void golden_free(golden_file *file) {
    if (file == NULL) {
        return;
    }
    if (file->tensors != NULL) {
        for (uint32_t i = 0; i < file->count; i++) {
            free(file->tensors[i].data);
        }
        free(file->tensors);
    }
    file->tensors = NULL;
    file->count = 0;
}
