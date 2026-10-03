/*
 * test_ops_elementwise.c - tests for unary and binary elementwise ops
 * (include/mlc/ops.h): values against PyTorch, broadcasting, non-contiguous
 * inputs, scalars, size-0 tensors, and in-place ops on views.
 *
 * Golden data: tests/ref/gen_ops_elementwise.py. To regenerate it:
 *     uv run --project tests/ref python tests/ref/gen_ops_elementwise.py
 *
 * Not tested here: inputs that must abort (shapes that do not broadcast, an
 * in-place output with stride 0 or one that would broadcast up). The harness
 * has no death tests.
 *
 * Tolerances: neg/abs/relu/maximum/minimum and add/sub/mul are exact in f32. Other ops
 * use library math (exp, erf, ...), which can differ from PyTorch in the last
 * bits, so they use atol 1e-6 and rtol 1e-5.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "golden.h"
#include "mlc/ops.h"
#include "mlc_test.h"
#include "tensor_check.h"

#define DATA(file) MLC_TEST_DATA_DIR "/" file

typedef mlc_tensor* (*unary_fn)(const mlc_tensor*);
typedef mlc_status (*unary_inplace_fn)(mlc_tensor*);
typedef mlc_tensor* (*binary_fn)(const mlc_tensor*, const mlc_tensor*);
typedef mlc_status (*binary_inplace_fn)(mlc_tensor*, const mlc_tensor*);

/* Reports a failure (without stopping the test) if an in-place op does not
 * return MLC_SUCCESS. */
#define EXPECT_OK(call)                                                \
    do {                                                               \
        const mlc_status st_ = (call);                                 \
        if (st_ != MLC_SUCCESS) {                                      \
            mlc_test_fail(__FILE__, __LINE__, "%s returned %s", #call, \
                          mlc_status_str(st_));                        \
        }                                                              \
    } while (0)

typedef struct {
    const char* name;
    unary_fn fn;
    unary_inplace_fn fn_;
    const char* input;   /* golden input: "x" or "xp" (positive) */
    const char* special; /* golden edge-case input */
    double atol;
    double rtol;
} unary_case;

static const unary_case UNARY[] = {
    {"neg", mlc_neg, mlc_neg_, "x", "special", 0.0, 0.0},
    {"abs", mlc_abs, mlc_abs_, "x", "special", 0.0, 0.0},
    {"relu", mlc_relu, mlc_relu_, "x", "special", 0.0, 0.0},
    {"exp", mlc_exp, mlc_exp_, "x", "special", 1e-6, 1e-5},
    {"tanh", mlc_tanh, mlc_tanh_, "x", "special", 1e-6, 1e-5},
    {"sigmoid", mlc_sigmoid, mlc_sigmoid_, "x", "special", 1e-6, 1e-5},
    {"gelu", mlc_gelu, mlc_gelu_, "x", "special", 1e-6, 1e-5},
    {"log", mlc_log, mlc_log_, "xp", "special_pos", 1e-6, 1e-5},
    {"sqrt", mlc_sqrt, mlc_sqrt_, "xp", "special_pos", 1e-6, 1e-5},
};
#define N_UNARY (sizeof UNARY / sizeof UNARY[0])

typedef struct {
    const char* name;
    binary_fn fn;
    binary_inplace_fn fn_;
    double atol;
    double rtol;
} binary_case;

static const binary_case BINARY[] = {
    {"add", mlc_add, mlc_add_, 0.0, 0.0},
    {"sub", mlc_sub, mlc_sub_, 0.0, 0.0},
    {"mul", mlc_mul, mlc_mul_, 0.0, 0.0},
    /* Division computed in double, then rounded to float, can differ by 1 ulp
     * from a float division. */
    {"div", mlc_div, mlc_div_, 0.0, 1e-6},
    {"maximum", mlc_maximum, mlc_maximum_, 0.0, 0.0},
    {"minimum", mlc_minimum, mlc_minimum_, 0.0, 0.0},
};
#define N_BINARY (sizeof BINARY / sizeof BINARY[0])

static int load(golden_file* gf) {
    char err[256] = {0};
    if (golden_load(DATA("ops_elementwise.mlct"), gf, err, sizeof err) != 0) {
        mlc_test_fail(__FILE__, __LINE__, "golden_load failed: %s", err);
        return 0;
    }
    return 1;
}

/* Builds "<op>_<suffix>" into `buf`. */
static const char* key(char* buf, size_t len, const char* op,
                       const char* suffix) {
    snprintf(buf, len, "%s_%s", op, suffix);
    return buf;
}

/* ---------------------------------------------------------------- unary -- */

/* Every unary op on a contiguous input, a transposed input, and edge cases.
 * Reports every failing case, not only the first. Inputs must not change. */
MLC_TEST(unary_ops_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    char k[64];
    for (size_t i = 0; i < N_UNARY; ++i) {
        const unary_case* c = &UNARY[i];
        const char* tsuffix = c->input[1] == 'p' ? "xpT" : "xT";

        mlc_tensor* x = tc_from_golden(&gf, c->input);
        mlc_tensor* r = c->fn(x);
        (void)tc_check_golden(r, &gf, key(k, sizeof k, c->name, c->input),
                              c->atol, c->rtol);
        (void)tc_check_golden(x, &gf, c->input, 0.0, 0.0); /* unchanged */
        mlc_tensor_free(r);

        mlc_tensor* xt = mlc_transpose(x, 0, 1);
        r = c->fn(xt);
        (void)tc_check_golden(r, &gf, key(k, sizeof k, c->name, tsuffix),
                              c->atol, c->rtol);
        mlc_tensor_free(r);
        mlc_tensor_free(xt);
        mlc_tensor_free(x);

        mlc_tensor* s = tc_from_golden(&gf, c->special);
        r = c->fn(s);
        (void)tc_check_golden(r, &gf, key(k, sizeof k, c->name, c->special),
                              c->atol, c->rtol);
        mlc_tensor_free(r);
        mlc_tensor_free(s);
    }
    golden_free(&gf);
}

/* In-place unary ops on a contiguous tensor and on a transposed view. */
MLC_TEST(unary_inplace_ops_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    char k[64];
    for (size_t i = 0; i < N_UNARY; ++i) {
        const unary_case* c = &UNARY[i];
        const char* tsuffix = c->input[1] == 'p' ? "xpT" : "xT";

        mlc_tensor* x = tc_from_golden(&gf, c->input);
        EXPECT_OK(c->fn_(x));
        (void)tc_check_golden(x, &gf, key(k, sizeof k, c->name, c->input),
                              c->atol, c->rtol);
        mlc_tensor_free(x);

        x = tc_from_golden(&gf, c->input);
        mlc_tensor* xt = mlc_transpose(x, 0, 1);
        EXPECT_OK(c->fn_(xt));
        (void)tc_check_golden(xt, &gf, key(k, sizeof k, c->name, tsuffix),
                              c->atol, c->rtol);
        mlc_tensor_free(xt);
        mlc_tensor_free(x);
    }
    golden_free(&gf);
}

/* exp_ on every second element changes only those elements of the base. */
MLC_TEST(unary_inplace_on_strided_slice_changes_base) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* v = tc_from_golden(&gf, "v10");
    mlc_tensor* even = mlc_slice(v, 0, 0, 10, 2);
    EXPECT_OK(mlc_exp_(even));
    MLC_ASSERT(tc_check_golden(v, &gf, "exp_inplace_step2", 1e-6, 1e-5));
    mlc_tensor_free(even);
    mlc_tensor_free(v);
    golden_free(&gf);
}

/* --------------------------------------------------------------- binary -- */

/* Every binary op: same shape (fast path), broadcast [3,1,4] x [5,4], a 0-dim
 * scalar on either side, and a transposed (non-contiguous) left input. */
MLC_TEST(binary_ops_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* a = tc_from_golden(&gf, "a_same");
    mlc_tensor* b = tc_from_golden(&gf, "b_same");
    mlc_tensor* abc = tc_from_golden(&gf, "a_bc");
    mlc_tensor* bbc = tc_from_golden(&gf, "b_bc");
    mlc_tensor* s = tc_from_golden(&gf, "s");
    mlc_tensor* ct = tc_from_golden(&gf, "c_t");
    mlc_tensor* at = mlc_transpose(a, 0, 1);
    char k[64];

    for (size_t i = 0; i < N_BINARY; ++i) {
        const binary_case* c = &BINARY[i];
        struct {
            mlc_tensor* lhs;
            mlc_tensor* rhs;
            const char* suffix;
        } cases[5] = {{a, b, "same"},
                      {abc, bbc, "bc"},
                      {a, s, "scalar_right"},
                      {s, b, "scalar_left"},
                      {at, ct, "T"}};
        for (int j = 0; j < 5; ++j) {
            mlc_tensor* r = c->fn(cases[j].lhs, cases[j].rhs);
            (void)tc_check_golden(r, &gf,
                                  key(k, sizeof k, c->name, cases[j].suffix),
                                  c->atol, c->rtol);
            mlc_tensor_free(r);
        }
    }
    /* No op may change its inputs. */
    MLC_ASSERT(tc_check_golden(a, &gf, "a_same", 0.0, 0.0));
    MLC_ASSERT(tc_check_golden(b, &gf, "b_same", 0.0, 0.0));

    mlc_tensor_free(at);
    mlc_tensor_free(ct);
    mlc_tensor_free(s);
    mlc_tensor_free(bbc);
    mlc_tensor_free(abc);
    mlc_tensor_free(b);
    mlc_tensor_free(a);
    golden_free(&gf);
}

/* In-place binary ops: same shape, scalar right side, and a transposed view
 * as the output. (The [3,1,4] x [5,4] case would broadcast the output up, so
 * it is not valid in place.) */
MLC_TEST(binary_inplace_ops_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* b = tc_from_golden(&gf, "b_same");
    mlc_tensor* s = tc_from_golden(&gf, "s");
    mlc_tensor* ct = tc_from_golden(&gf, "c_t");
    char k[64];

    for (size_t i = 0; i < N_BINARY; ++i) {
        const binary_case* c = &BINARY[i];

        mlc_tensor* a = tc_from_golden(&gf, "a_same");
        EXPECT_OK(c->fn_(a, b));
        (void)tc_check_golden(a, &gf, key(k, sizeof k, c->name, "same"),
                              c->atol, c->rtol);
        mlc_tensor_free(a);

        a = tc_from_golden(&gf, "a_same");
        EXPECT_OK(c->fn_(a, s));
        (void)tc_check_golden(a, &gf, key(k, sizeof k, c->name, "scalar_right"),
                              c->atol, c->rtol);
        mlc_tensor_free(a);

        a = tc_from_golden(&gf, "a_same");
        mlc_tensor* at = mlc_transpose(a, 0, 1);
        EXPECT_OK(c->fn_(at, ct));
        (void)tc_check_golden(at, &gf, key(k, sizeof k, c->name, "T"), c->atol,
                              c->rtol);
        mlc_tensor_free(at);
        mlc_tensor_free(a);
    }
    MLC_ASSERT(tc_check_golden(b, &gf, "b_same", 0.0, 0.0)); /* unchanged */

    mlc_tensor_free(ct);
    mlc_tensor_free(s);
    mlc_tensor_free(b);
    golden_free(&gf);
}

/* add_ through a [4, 1] column slice changes column 2 of the base only. */
MLC_TEST(inplace_through_column_slice_changes_base) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* a = tc_from_golden(&gf, "a_same");
    mlc_tensor* col = mlc_slice(a, 1, 2, 3, 1);
    mlc_tensor* one = mlc_full(NULL, 0, MLC_F32, 1.0);
    EXPECT_OK(mlc_add_(col, one));
    MLC_ASSERT(tc_check_golden(a, &gf, "add_inplace_column", 0.0, 0.0));
    mlc_tensor_free(one);
    mlc_tensor_free(col);
    mlc_tensor_free(a);
    golden_free(&gf);
}

/* a [4, 6] += b [6]: the right side broadcasts over the rows. */
MLC_TEST(inplace_with_broadcast_right_side) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* a = tc_from_golden(&gf, "a_same");
    mlc_tensor* row = tc_from_golden(&gf, "b_row");
    EXPECT_OK(mlc_add_(a, row));
    MLC_ASSERT(tc_check_golden(a, &gf, "add_inplace_bc", 0.0, 0.0));
    mlc_tensor_free(row);
    mlc_tensor_free(a);
    golden_free(&gf);
}

/* sq += sq.T: the right side shares storage with the output in a different
 * layout, so a naive loop reads elements that it already overwrote. */
MLC_TEST(inplace_with_overlapping_input) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* sq = tc_from_golden(&gf, "sq");
    mlc_tensor* sqt = mlc_transpose(sq, 0, 1);
    EXPECT_OK(mlc_add_(sq, sqt));
    MLC_ASSERT(tc_check_golden(sq, &gf, "add_inplace_overlap", 0.0, 0.0));
    mlc_tensor_free(sqt);
    mlc_tensor_free(sq);
    golden_free(&gf);
}

/* v6[1:5] += v6[0:4]: same shape and strides, but the right side starts one
 * element earlier in the same storage. Comparing only shape and strides misses
 * this overlap; a forward loop then reads elements it already overwrote. */
MLC_TEST(inplace_with_shifted_overlapping_input) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* v = tc_from_golden(&gf, "v6");
    mlc_tensor* dst = mlc_slice(v, 0, 1, 5, 1);
    mlc_tensor* src = mlc_slice(v, 0, 0, 4, 1);
    EXPECT_OK(mlc_add_(dst, src));
    MLC_ASSERT(
        tc_check_golden(v, &gf, "add_inplace_shifted_overlap", 0.0, 0.0));
    mlc_tensor_free(src);
    mlc_tensor_free(dst);
    mlc_tensor_free(v);
    golden_free(&gf);
}

/* ------------------------------------------------------------------ pow -- */

/* mlc_pow(a, b) with tensor exponents (same shape and broadcast), and with
 * 0-dim scalar exponents 2, 2.5, -0.5, and 0.5 (negative base -> nan). */
MLC_TEST(pow_ops_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    const double atol = 1e-6;
    const double rtol = 1e-5;

    const char* tensor_cases[2][3] = {{"pa_same", "pb_same", "pow_same"},
                                      {"pa_bc", "pb_bc", "pow_bc"}};
    for (int i = 0; i < 2; ++i) {
        mlc_tensor* pa = tc_from_golden(&gf, tensor_cases[i][0]);
        mlc_tensor* pb = tc_from_golden(&gf, tensor_cases[i][1]);
        mlc_tensor* r = mlc_pow(pa, pb);
        (void)tc_check_golden(r, &gf, tensor_cases[i][2], atol, rtol);
        mlc_tensor_free(r);
        mlc_tensor_free(pb);
        mlc_tensor_free(pa);
    }

    struct {
        const char* input;
        float exponent;
        const char* expected;
    } scalar_cases[4] = {{"x", 2.0f, "pow_2_x"},
                         {"xp", 2.5f, "pow_2_5_xp"},
                         {"xp", -0.5f, "pow_neg_half_xp"},
                         {"special", 0.5f, "pow_0_5_special"}};
    for (int i = 0; i < 4; ++i) {
        mlc_tensor* x = tc_from_golden(&gf, scalar_cases[i].input);
        mlc_tensor* e = mlc_full(NULL, 0, MLC_F32, scalar_cases[i].exponent);
        mlc_tensor* r = mlc_pow(x, e);
        (void)tc_check_golden(r, &gf, scalar_cases[i].expected, atol, rtol);
        mlc_tensor_free(r);
        EXPECT_OK(mlc_pow_(x, e));
        (void)tc_check_golden(x, &gf, scalar_cases[i].expected, atol, rtol);
        mlc_tensor_free(e);
        mlc_tensor_free(x);
    }

    mlc_tensor* pa = tc_from_golden(&gf, "pa_same");
    mlc_tensor* pb = tc_from_golden(&gf, "pb_same");
    EXPECT_OK(mlc_pow_(pa, pb));
    (void)tc_check_golden(pa, &gf, "pow_same", atol, rtol);
    mlc_tensor_free(pb);
    mlc_tensor_free(pa);
    golden_free(&gf);
}

/* ------------------------------------------------------- shapes, memory -- */

MLC_TEST(result_is_new_contiguous_tensor) {
    const int64_t shape[2] = {2, 3};
    mlc_tensor* a = mlc_ones(shape, 2, MLC_F32);
    mlc_tensor* b = mlc_ones(shape, 2, MLC_F32);
    mlc_tensor* r = mlc_add(a, b);
    MLC_ASSERT(tc_check_shape(r, shape, 2));
    MLC_ASSERT(r->data != a->data && r->data != b->data);
    MLC_ASSERT(mlc_is_contiguous(r));
    MLC_ASSERT_EQ_INT(r->data->ref_count, 1);
    MLC_ASSERT_EQ_INT(a->data->ref_count, 1); /* no leaked references */
    mlc_tensor* u = mlc_exp(a);
    MLC_ASSERT(u != NULL && u->data != a->data && mlc_is_contiguous(u));
    mlc_tensor_free(u);
    mlc_tensor_free(r);
    mlc_tensor_free(b);
    mlc_tensor_free(a);
}

MLC_TEST(scalar_and_zero_size_shapes) {
    mlc_tensor* s1 = mlc_full(NULL, 0, MLC_F32, 2.0);
    mlc_tensor* s2 = mlc_full(NULL, 0, MLC_F32, 3.0);
    mlc_tensor* r = mlc_mul(s1, s2);
    MLC_ASSERT(tc_check_shape(r, NULL, 0));
    MLC_ASSERT(*(const float*)mlc_tensor_data(r) == 6.0f);
    mlc_tensor_free(r);

    /* [0, 4] + [4] -> [0, 4] */
    mlc_tensor* z = mlc_zeros((int64_t[]){0, 4}, 2, MLC_F32);
    mlc_tensor* row = mlc_ones((int64_t[]){4}, 1, MLC_F32);
    r = mlc_add(z, row);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){0, 4}, 2));
    mlc_tensor_free(r);
    r = mlc_relu(z);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){0, 4}, 2));
    mlc_tensor_free(r);
    EXPECT_OK(mlc_add_(z, row)); /* no elements: must not crash */

    mlc_tensor_free(row);
    mlc_tensor_free(z);
    mlc_tensor_free(s2);
    mlc_tensor_free(s1);
}

/* An expanded (stride 0) input is valid on the right side. */
MLC_TEST(expanded_input_on_right_side) {
    mlc_tensor* a = mlc_arange(0, 6, 1, MLC_F32);
    mlc_tensor* m = mlc_view(a, (int64_t[]){2, 3}, 2);
    mlc_tensor* col =
        mlc_from_data((float[]){10, 20}, (int64_t[]){2, 1}, 2, MLC_F32);
    mlc_tensor* e = mlc_expand(col, (int64_t[]){2, 3}, 2);
    mlc_tensor* r = mlc_add(m, e);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){2, 3}, 2));
    const float expected[6] = {10, 11, 12, 23, 24, 25};
    float* got = tc_gather_f32(r);
    MLC_ASSERT_ALLCLOSE_F32(got, expected, 6, 0.0, 0.0);
    free(got);
    mlc_tensor_free(r);
    mlc_tensor_free(e);
    mlc_tensor_free(col);
    mlc_tensor_free(m);
    mlc_tensor_free(a);
}
