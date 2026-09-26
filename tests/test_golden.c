/*
 * test_golden.c - self-test for the golden file reader (tests/support/golden.c).
 *
 * Generate the data first:
 *     uv run --project tests/ref python tests/ref/gen_golden_selftest.py
 *
 * This file has its own main() and small check macros. Convert it to the shared test
 * harness when the harness exists (TODO.md, Phase 0).
 */
#include "golden.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, \
                    #cond);                                                  \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

#define DATA(file) MLC_TEST_DATA_DIR "/" file

static void test_valid_file(void) {
    golden_file gf;
    char err[256] = {0};
    const int rc = golden_load(DATA("golden_selftest.mlct"), &gf, err, sizeof err);
    if (rc != 0) {
        fprintf(stderr, "golden_load failed: %s\n", err);
        g_failures++;
        return;
    }
    CHECK(gf.count == 7);

    const golden_tensor *t = golden_require(&gf, "scalar_f32", GOLDEN_F32);
    CHECK(t->ndim == 0 && t->numel == 1);
    CHECK(((const float *)t->data)[0] == 3.5f);

    t = golden_require(&gf, "arange_2x3_f32", GOLDEN_F32);
    CHECK(t->ndim == 2 && t->shape[0] == 2 && t->shape[1] == 3 && t->numel == 6);
    for (int64_t i = 0; i < t->numel; i++) {
        CHECK(((const float *)t->data)[i] == 0.5f * (float)i);
    }
    CHECK((uintptr_t)t->data % 64 == 0);

    t = golden_require(&gf, "f64", GOLDEN_F64);
    CHECK(t->numel == 2);
    CHECK(((const double *)t->data)[0] == 1e-300);
    CHECK(((const double *)t->data)[1] == 3.141592653589793); /* np.pi; M_PI is not C11 */

    t = golden_require(&gf, "i32", GOLDEN_I32);
    CHECK(((const int32_t *)t->data)[0] == -1);
    CHECK(((const int32_t *)t->data)[1] == INT32_MAX);

    t = golden_require(&gf, "i64", GOLDEN_I64);
    CHECK(((const int64_t *)t->data)[0] == -(INT64_C(1) << 62));
    CHECK(((const int64_t *)t->data)[1] == (INT64_C(1) << 40));

    t = golden_require(&gf, "empty_0x4_f32", GOLDEN_F32);
    CHECK(t->ndim == 2 && t->shape[0] == 0 && t->shape[1] == 4 && t->numel == 0);
    CHECK(t->data != NULL);

    /* Source was arange(6).reshape(2, 3).T, so row-major data is 0 3 1 4 2 5. */
    t = golden_require(&gf, "transposed_3x2_f32", GOLDEN_F32);
    CHECK(t->shape[0] == 3 && t->shape[1] == 2);
    const float expected_t[6] = {0, 3, 1, 4, 2, 5};
    CHECK(memcmp(t->data, expected_t, sizeof expected_t) == 0);

    CHECK(golden_get(&gf, "missing") == NULL);

    golden_free(&gf);
    CHECK(gf.count == 0 && gf.tensors == NULL);
    golden_free(&gf); /* A second free must be safe. */
}

/* Each invalid file must fail with a message, and leave `gf` empty. */
static void expect_load_error(const char *path, const char *expected_substring) {
    golden_file gf;
    char err[256] = {0};
    CHECK(golden_load(path, &gf, err, sizeof err) == -1);
    CHECK(gf.count == 0 && gf.tensors == NULL);
    if (strstr(err, expected_substring) == NULL) {
        fprintf(stderr, "%s: error '%s' does not contain '%s'\n",
                path, err, expected_substring);
        g_failures++;
    }
}

static void test_invalid_files(void) {
    expect_load_error(DATA("does_not_exist.mlct"), "cannot open");
    expect_load_error(DATA("golden_bad_magic.mlct"), "bad magic");
    expect_load_error(DATA("golden_truncated.mlct"), "truncated");
    expect_load_error(DATA("golden_trailing.mlct"), "after the last entry");
}

int main(void) {
    test_valid_file();
    test_invalid_files();
    if (g_failures != 0) {
        fprintf(stderr, "test_golden: %d check(s) failed\n", g_failures);
        return 1;
    }
    printf("test_golden: OK\n");
    return 0;
}
