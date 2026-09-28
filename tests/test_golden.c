/*
 * test_golden.c - self-test for the golden file reader
 * (tests/support/golden.c).
 *
 * The data comes from tests/ref/gen_golden_selftest.py. To regenerate it:
 *     uv run --project tests/ref python tests/ref/gen_golden_selftest.py
 *
 * Note: if an assertion fails, the test returns early and does not free the
 * golden_file. This leak happens only in a failed test.
 */
#include <stdint.h>
#include <string.h>

#include "golden.h"
#include "mlc_test.h"

#define DATA(file) MLC_TEST_DATA_DIR "/" file

MLC_TEST(golden_load_valid_file) {
    golden_file gf;
    char err[256] = {0};
    MLC_ASSERT_MSG(
        golden_load(DATA("golden_selftest.mlct"), &gf, err, sizeof err) == 0,
        "golden_load failed: %s", err);
    MLC_ASSERT_EQ_INT(gf.count, 7);
    MLC_ASSERT(golden_get(&gf, "missing") == NULL);
    golden_free(&gf);
}

MLC_TEST(golden_scalar_and_float_values) {
    golden_file gf;
    char err[256] = {0};
    MLC_ASSERT_MSG(
        golden_load(DATA("golden_selftest.mlct"), &gf, err, sizeof err) == 0,
        "golden_load failed: %s", err);

    const golden_tensor* t = golden_require(&gf, "scalar_f32", GOLDEN_F32);
    MLC_ASSERT_EQ_INT(t->ndim, 0);
    MLC_ASSERT_EQ_INT(t->numel, 1);
    MLC_ASSERT(((const float*)t->data)[0] == 3.5f);

    t = golden_require(&gf, "arange_2x3_f32", GOLDEN_F32);
    MLC_ASSERT_EQ_INT(t->ndim, 2);
    MLC_ASSERT_EQ_INT(t->shape[0], 2);
    MLC_ASSERT_EQ_INT(t->shape[1], 3);
    const float expected_arange[6] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 2.5f};
    /* Exact values: tolerance 0. */
    MLC_ASSERT_ALLCLOSE_F32((const float*)t->data, expected_arange, 6, 0.0,
                            0.0);
    MLC_ASSERT_MSG((uintptr_t)t->data % 64 == 0, "data is not 64-byte aligned");

    t = golden_require(&gf, "f64", GOLDEN_F64);
    /* 3.141592653589793 is np.pi; M_PI is not part of C11. */
    const double expected_f64[2] = {1e-300, 3.141592653589793};
    MLC_ASSERT_ALLCLOSE_F64((const double*)t->data, expected_f64, 2, 0.0, 0.0);

    /* Source was arange(6).reshape(2, 3).T, so row-major data is 0 3 1 4 2 5.
     */
    t = golden_require(&gf, "transposed_3x2_f32", GOLDEN_F32);
    MLC_ASSERT_EQ_INT(t->shape[0], 3);
    MLC_ASSERT_EQ_INT(t->shape[1], 2);
    const float expected_t[6] = {0, 3, 1, 4, 2, 5};
    MLC_ASSERT_ALLCLOSE_F32((const float*)t->data, expected_t, 6, 0.0, 0.0);

    golden_free(&gf);
}

MLC_TEST(golden_integer_values) {
    golden_file gf;
    char err[256] = {0};
    MLC_ASSERT_MSG(
        golden_load(DATA("golden_selftest.mlct"), &gf, err, sizeof err) == 0,
        "golden_load failed: %s", err);

    const golden_tensor* t = golden_require(&gf, "i32", GOLDEN_I32);
    MLC_ASSERT_EQ_INT(((const int32_t*)t->data)[0], -1);
    MLC_ASSERT_EQ_INT(((const int32_t*)t->data)[1], INT32_MAX);

    t = golden_require(&gf, "i64", GOLDEN_I64);
    MLC_ASSERT_EQ_INT(((const int64_t*)t->data)[0], -(INT64_C(1) << 62));
    MLC_ASSERT_EQ_INT(((const int64_t*)t->data)[1], INT64_C(1) << 40);

    golden_free(&gf);
}

MLC_TEST(golden_empty_tensor) {
    golden_file gf;
    char err[256] = {0};
    MLC_ASSERT_MSG(
        golden_load(DATA("golden_selftest.mlct"), &gf, err, sizeof err) == 0,
        "golden_load failed: %s", err);

    const golden_tensor* t = golden_require(&gf, "empty_0x4_f32", GOLDEN_F32);
    MLC_ASSERT_EQ_INT(t->ndim, 2);
    MLC_ASSERT_EQ_INT(t->shape[0], 0);
    MLC_ASSERT_EQ_INT(t->shape[1], 4);
    MLC_ASSERT_EQ_INT(t->numel, 0);
    MLC_ASSERT(t->data != NULL);

    golden_free(&gf);
}

MLC_TEST(golden_free_twice_is_safe) {
    golden_file gf;
    MLC_ASSERT(golden_load(DATA("golden_selftest.mlct"), &gf, NULL, 0) == 0);
    golden_free(&gf);
    MLC_ASSERT(gf.count == 0 && gf.tensors == NULL);
    golden_free(&gf);
}

/* Each invalid file must fail with a message that contains `expected`, and
 * leave the golden_file empty. Returns 1 on success, so the test can stop at
 * the first failure. */
static int load_fails_with(const char* path, const char* expected) {
    golden_file gf;
    char err[256] = {0};
    if (golden_load(path, &gf, err, sizeof err) != -1) {
        mlc_test_fail(__FILE__, __LINE__, "%s: golden_load did not fail", path);
        golden_free(&gf);
        return 0;
    }
    if (gf.count != 0 || gf.tensors != NULL) {
        mlc_test_fail(__FILE__, __LINE__,
                      "%s: golden_file is not empty after failure", path);
        return 0;
    }
    if (strstr(err, expected) == NULL) {
        mlc_test_fail(__FILE__, __LINE__,
                      "%s: error '%s' does not contain '%s'", path, err,
                      expected);
        return 0;
    }
    return 1;
}

MLC_TEST(golden_invalid_files) {
    MLC_ASSERT(load_fails_with(DATA("does_not_exist.mlct"), "cannot open"));
    MLC_ASSERT(load_fails_with(DATA("golden_bad_magic.mlct"), "bad magic"));
    MLC_ASSERT(load_fails_with(DATA("golden_truncated.mlct"), "truncated"));
    MLC_ASSERT(
        load_fails_with(DATA("golden_trailing.mlct"), "after the last entry"));
}
