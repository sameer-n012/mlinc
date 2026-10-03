/*
 * tensor_check.h - test helpers for mlc_tensor values and shapes.
 *
 * The helpers read tensor elements with their own strided walk (they use only
 * shape, strides, offset, and storage), so tests do not depend on
 * mlc_contiguous or other library code being correct.
 *
 * All check_* functions report a failure with mlc_test_fail and return 0, or
 * return 1 on success. Use them as MLC_ASSERT(check_...(...)).
 */
#ifndef MLC_TESTS_TENSOR_CHECK_H
#define MLC_TESTS_TENSOR_CHECK_H

#include <stdint.h>

#include "golden.h"
#include "mlc/tensor.h"

/* Returns the f32 element of `t` at the multi-index `idx`. */
float tc_at(const mlc_tensor* t, const int64_t* idx);

/* Copies all elements of `t` in row-major order into a new malloc'd array.
 * The caller frees it. Returns NULL if numel is 0 or malloc fails. */
float* tc_gather_f32(const mlc_tensor* t);

/* Creates a new f32 tensor with the shape and values of a golden entry.
 * Aborts (via golden_require) if the entry does not exist or is not f32. */
mlc_tensor* tc_from_golden(const golden_file* gf, const char* name);

/* Checks that `t` is not NULL and has exactly this shape. */
int tc_check_shape(const mlc_tensor* t, const int64_t* shape, int64_t ndim);

/* Checks shape and values of `t` against the golden entry `name`. Tolerances
 * follow numpy.isclose (see mlc_test.h). */
int tc_check_golden(const mlc_tensor* t, const golden_file* gf,
                    const char* name, double atol, double rtol);

#endif /* MLC_TESTS_TENSOR_CHECK_H */
