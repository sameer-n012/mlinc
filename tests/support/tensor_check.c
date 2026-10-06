/*
 * tensor_check.c - test helpers for mlc_tensor values and shapes.
 * See tensor_check.h.
 */
#include "tensor_check.h"

#include <stdlib.h>

#include "mlc_test.h"

float tc_at(const mlc_tensor* t, const int64_t* idx) {
    int64_t pos = t->offset;
    for (int64_t d = 0; d < t->ndim; ++d) {
        pos += idx[d] * t->strides[d];
    }
    return ((const float*)t->data->data)[pos];
}

float* tc_gather_f32(const mlc_tensor* t) {
    const int64_t n = mlc_tensor_numel(t);
    float* out = n > 0 ? malloc((size_t)n * sizeof(float)) : NULL;
    if (out == NULL) {
        return NULL;
    }
    int64_t idx[MLC_MAX_DIMS] = {0};
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tc_at(t, idx);
        /* Increment the multi-index like an odometer, last dim fastest. */
        for (int64_t d = t->ndim - 1; d >= 0; --d) {
            if (++idx[d] < t->shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
    return out;
}

mlc_tensor* tc_from_golden(const golden_file* gf, const char* name) {
    const golden_tensor* g = golden_require(gf, name, GOLDEN_F32);
    return mlc_from_data(g->data, g->shape, (int64_t)g->ndim, MLC_F32);
}

int tc_check_shape(const mlc_tensor* t, const int64_t* shape, int64_t ndim) {
    if (t == NULL) {
        mlc_test_fail(__FILE__, __LINE__, "tensor is NULL");
        return 0;
    }
    if (t->ndim != ndim) {
        mlc_test_fail(__FILE__, __LINE__, "ndim: got %lld, expected %lld",
                      (long long)t->ndim, (long long)ndim);
        return 0;
    }
    for (int64_t i = 0; i < ndim; ++i) {
        if (t->shape[i] != shape[i]) {
            mlc_test_fail(__FILE__, __LINE__,
                          "dim %lld: got %lld, expected %lld", (long long)i,
                          (long long)t->shape[i], (long long)shape[i]);
            return 0;
        }
    }
    return 1;
}

int tc_check_golden(const mlc_tensor* t, const golden_file* gf,
                    const char* name, double atol, double rtol) {
    const golden_tensor* g = golden_require(gf, name, GOLDEN_F32);
    return tc_check_golden_values(t, gf, name, g->shape, (int64_t)g->ndim, atol,
                                  rtol);
}

int tc_check_golden_values(const mlc_tensor* t, const golden_file* gf,
                           const char* name, const int64_t* shape, int64_t ndim,
                           double atol, double rtol) {
    const golden_tensor* g = golden_require(gf, name, GOLDEN_F32);
    if (!tc_check_shape(t, shape, ndim)) {
        mlc_test_fail(__FILE__, __LINE__, "case '%s': wrong shape", name);
        return 0;
    }
    if (t->dtype != MLC_F32) {
        mlc_test_fail(__FILE__, __LINE__, "case '%s': dtype %s, expected %s",
                      name, mlc_dtype_str(t->dtype), mlc_dtype_str(MLC_F32));
        return 0;
    }
    float* values = tc_gather_f32(t);
    const int ok = mlc_test_check_allclose_f32(
        __FILE__, __LINE__, name, "golden", values, (const float*)g->data,
        (size_t)g->numel, atol, rtol);
    free(values);
    return ok;
}

int64_t tc_at_i64(const mlc_tensor* t, const int64_t* idx) {
    int64_t pos = t->offset;
    for (int64_t d = 0; d < t->ndim; ++d) {
        pos += idx[d] * t->strides[d];
    }
    return ((const int64_t*)t->data->data)[pos];
}

int tc_check_golden_i64(const mlc_tensor* t, const golden_file* gf,
                        const char* name, const int64_t* shape, int64_t ndim) {
    const golden_tensor* g = golden_require(gf, name, GOLDEN_I64);
    if (shape == NULL) {
        shape = g->shape;
        ndim = (int64_t)g->ndim;
    }
    if (!tc_check_shape(t, shape, ndim)) {
        mlc_test_fail(__FILE__, __LINE__, "case '%s': wrong shape", name);
        return 0;
    }
    if (t->dtype != MLC_I64) {
        mlc_test_fail(__FILE__, __LINE__, "case '%s': dtype %s, expected %s",
                      name, mlc_dtype_str(t->dtype), mlc_dtype_str(MLC_I64));
        return 0;
    }
    const int64_t* expected = (const int64_t*)g->data;
    int64_t idx[MLC_MAX_DIMS] = {0};
    for (int64_t i = 0; i < g->numel; ++i) {
        const int64_t got = tc_at_i64(t, idx);
        if (got != expected[i]) {
            mlc_test_fail(__FILE__, __LINE__,
                          "case '%s': element %lld is %lld, expected %lld",
                          name, (long long)i, (long long)got,
                          (long long)expected[i]);
            return 0;
        }
        for (int64_t d = t->ndim - 1; d >= 0; --d) {
            if (++idx[d] < t->shape[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
    return 1;
}
