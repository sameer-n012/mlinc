/*
 * test_tensor_create.c - tests for tensor creation, metadata, and
 * mlc_broadcast_shapes (include/mlc/tensor.h).
 */
#include <stdint.h>
#include <string.h>

#include "mlc/tensor.h"
#include "mlc_test.h"

/* Checks shape, contiguous strides, offset 0, and alignment of a new tensor. */
static int check_new_tensor(const mlc_tensor* t, const int64_t* shape,
                            int64_t ndim) {
    if (t == NULL) {
        mlc_test_fail(__FILE__, __LINE__, "tensor is NULL");
        return 0;
    }
    if (t->ndim != ndim || t->offset != 0) {
        mlc_test_fail(__FILE__, __LINE__, "ndim/offset: got %lld/%lld",
                      (long long)t->ndim, (long long)t->offset);
        return 0;
    }
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        if (t->shape[i] != shape[i] || t->strides[i] != stride) {
            mlc_test_fail(
                __FILE__, __LINE__,
                "dim %lld: shape %lld stride %lld, expected %lld %lld",
                (long long)i, (long long)t->shape[i], (long long)t->strides[i],
                (long long)shape[i], (long long)stride);
            return 0;
        }
        stride *= shape[i];
    }
    if ((uintptr_t)mlc_tensor_data(t) % 64 != 0) {
        mlc_test_fail(__FILE__, __LINE__, "data is not 64-byte aligned");
        return 0;
    }
    return 1;
}

MLC_TEST(empty_shape_strides_numel) {
    const int64_t shape[3] = {2, 3, 4};
    mlc_tensor* t = mlc_empty(shape, 3, MLC_F32);
    MLC_ASSERT(check_new_tensor(t, shape, 3));
    MLC_ASSERT_EQ_INT(mlc_tensor_numel(t), 24);
    MLC_ASSERT(mlc_is_contiguous(t));
    MLC_ASSERT(t->dtype == MLC_F32);
    MLC_ASSERT_EQ_INT(t->data->ref_count, 1);
    MLC_ASSERT(t->data->size >= 24 * sizeof(float));
    mlc_tensor_free(t);
}

MLC_TEST(scalar_tensor) {
    mlc_tensor* t = mlc_full(NULL, 0, MLC_F32, 2.5);
    MLC_ASSERT(check_new_tensor(t, NULL, 0));
    MLC_ASSERT_EQ_INT(mlc_tensor_numel(t), 1);
    MLC_ASSERT(mlc_is_contiguous(t));
    MLC_ASSERT(*(const float*)mlc_tensor_data(t) == 2.5f);
    mlc_tensor_free(t);
}

MLC_TEST(zero_size_tensor) {
    const int64_t shape[2] = {0, 4};
    mlc_tensor* t = mlc_zeros(shape, 2, MLC_F32);
    MLC_ASSERT(check_new_tensor(t, shape, 2));
    MLC_ASSERT_EQ_INT(mlc_tensor_numel(t), 0);
    MLC_ASSERT(mlc_is_contiguous(t));
    mlc_tensor_free(t);
}

MLC_TEST(zeros_ones_full_values) {
    const int64_t shape[2] = {3, 5};
    const float expected_zero[15] = {0};
    float expected_one[15];
    float expected_full[15];
    for (int i = 0; i < 15; ++i) {
        expected_one[i] = 1.0f;
        expected_full[i] = -3.25f;
    }

    mlc_tensor* z = mlc_zeros(shape, 2, MLC_F32);
    mlc_tensor* o = mlc_ones(shape, 2, MLC_F32);
    mlc_tensor* f = mlc_full(shape, 2, MLC_F32, -3.25);
    MLC_ASSERT(check_new_tensor(z, shape, 2));
    MLC_ASSERT(check_new_tensor(o, shape, 2));
    MLC_ASSERT(check_new_tensor(f, shape, 2));
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(z), expected_zero, 15,
                            0.0, 0.0);
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(o), expected_one, 15,
                            0.0, 0.0);
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(f), expected_full, 15,
                            0.0, 0.0);
    mlc_tensor_free(z);
    mlc_tensor_free(o);
    mlc_tensor_free(f);
}

MLC_TEST(from_data_copies_source) {
    float src[6] = {1, 2, 3, 4, 5, 6};
    const float expected[6] = {1, 2, 3, 4, 5, 6};
    const int64_t shape[2] = {2, 3};
    mlc_tensor* t = mlc_from_data(src, shape, 2, MLC_F32);
    MLC_ASSERT(check_new_tensor(t, shape, 2));
    src[0] = 100.0f; /* the tensor owns a copy: it must not change */
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(t), expected, 6, 0.0,
                            0.0);
    mlc_tensor_free(t);
}

/* Checks the values of mlc_arange(start, end, step) against `expected`.
 * The values are compared with a small tolerance, because start + i * step is
 * computed in double and rounded to float. */
static int check_arange(double start, double end, double step,
                        const float* expected, int64_t n) {
    mlc_tensor* t = mlc_arange(start, end, step, MLC_F32);
    int ok = t != NULL && t->ndim == 1 && t->shape[0] == n;
    if (!ok) {
        mlc_test_fail(__FILE__, __LINE__,
                      "arange(%g, %g, %g): expected %lld elements, got %lld",
                      start, end, step, (long long)n,
                      t != NULL ? (long long)t->shape[0] : -1LL);
    } else {
        ok = mlc_test_check_allclose_f32(
            __FILE__, __LINE__, "arange", "expected",
            (const float*)mlc_tensor_data(t), expected, (size_t)n, 1e-6, 1e-6);
    }
    mlc_tensor_free(t);
    return ok;
}

MLC_TEST(arange_counts_and_values) {
    const float a[5] = {0, 1, 2, 3, 4};
    MLC_ASSERT(check_arange(0, 5, 1, a, 5));
    /* (1 - 0) / 0.3 = 3.33..., so ceil gives 4 elements. */
    const float b[4] = {0.0f, 0.3f, 0.6f, 0.9f};
    MLC_ASSERT(check_arange(0, 1, 0.3, b, 4));
    /* 0.3 / 0.1 is 2.9999999999999996 in double; PyTorch gives 3. */
    const float c[3] = {0.0f, 0.1f, 0.2f};
    MLC_ASSERT(check_arange(0, 0.3, 0.1, c, 3));
    const float d[5] = {5, 4, 3, 2, 1};
    MLC_ASSERT(check_arange(5, 0, -1, d, 5));
    MLC_ASSERT(check_arange(2, 2, 1, NULL, 0));
}

MLC_TEST(free_null_is_noop) { mlc_tensor_free(NULL); }

/* Calls mlc_broadcast_shapes and compares the result with `expected`.
 * `expected_ndim` < 0 means the shapes must NOT broadcast. */
static int check_broadcast(const int64_t* a, int64_t na, const int64_t* b,
                           int64_t nb, const int64_t* expected,
                           int64_t expected_ndim) {
    int64_t out[MLC_MAX_DIMS] = {0};
    int64_t out_ndim = -1;
    const bool ok = mlc_broadcast_shapes(a, na, b, nb, out, &out_ndim);
    if (expected_ndim < 0) {
        if (ok) {
            mlc_test_fail(__FILE__, __LINE__, "shapes broadcast, but must not");
        }
        return !ok;
    }
    if (!ok || out_ndim != expected_ndim) {
        mlc_test_fail(__FILE__, __LINE__, "ok=%d ndim=%lld, expected ndim %lld",
                      ok, (long long)out_ndim, (long long)expected_ndim);
        return 0;
    }
    for (int64_t i = 0; i < expected_ndim; ++i) {
        if (out[i] != expected[i]) {
            mlc_test_fail(__FILE__, __LINE__,
                          "dim %lld: got %lld, expected %lld", (long long)i,
                          (long long)out[i], (long long)expected[i]);
            return 0;
        }
    }
    return 1;
}

MLC_TEST(broadcast_shapes_rules) {
    MLC_ASSERT(check_broadcast((int64_t[]){3, 1}, 2, (int64_t[]){4}, 1,
                               (int64_t[]){3, 4}, 2));
    MLC_ASSERT(check_broadcast((int64_t[]){2, 3}, 2, (int64_t[]){3}, 1,
                               (int64_t[]){2, 3}, 2));
    MLC_ASSERT(check_broadcast((int64_t[]){5, 1, 4}, 3, (int64_t[]){3, 1}, 2,
                               (int64_t[]){5, 3, 4}, 3));
    /* A scalar broadcasts with every shape. */
    MLC_ASSERT(
        check_broadcast(NULL, 0, (int64_t[]){2, 3}, 2, (int64_t[]){2, 3}, 2));
    /* Size 0 wins over size 1. */
    MLC_ASSERT(check_broadcast((int64_t[]){0}, 1, (int64_t[]){1}, 1,
                               (int64_t[]){0}, 1));
    MLC_ASSERT(check_broadcast((int64_t[]){2}, 1, (int64_t[]){3}, 1, NULL, -1));
    MLC_ASSERT(
        check_broadcast((int64_t[]){2, 3}, 2, (int64_t[]){2}, 1, NULL, -1));
}

MLC_TEST(is_contiguous_ignores_size_one_strides) {
    const int64_t shape[3] = {2, 1, 3};
    mlc_tensor* t = mlc_zeros(shape, 3, MLC_F32);
    MLC_ASSERT(t != NULL);
    t->strides[1] = 999; /* never used, because the dim has size 1 */
    MLC_ASSERT(mlc_is_contiguous(t));
    mlc_tensor_free(t);
}
