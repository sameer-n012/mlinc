/*
 * test_ops_reduce.c - tests for reductions (include/mlc/ops.h): sum, prod,
 * mean, max, min, median, argmax, argmin over one dim, and sum, prod, mean,
 * max, min over several dims.
 *
 * Semantics follow PyTorch (see tests/ref/gen_ops_reduce.py):
 * - max/min/median propagate NaN; argmax/argmin return the index of the first
 *   NaN, else the first occurrence of the extreme value (ties -> lowest index).
 * - median of an even count is the LOWER of the two middle values.
 * - argmax/argmin return MLC_I64 tensors.
 * - Over a dim of size 0: sum = 0, prod = 1, mean = NaN.
 *
 * Not tested (they must abort; the harness has no death tests): max/min/
 * argmax/argmin/median over a dim of size 0, and dims out of range.
 *
 * Golden data: tests/ref/gen_ops_reduce.py. To regenerate it:
 *     uv run --project tests/ref python tests/ref/gen_ops_reduce.py
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "golden.h"
#include "mlc/ops.h"
#include "mlc_test.h"
#include "tensor_check.h"

#define DATA(file) MLC_TEST_DATA_DIR "/" file

typedef mlc_tensor* (*reduce_fn)(const mlc_tensor*, int64_t, bool);
typedef mlc_tensor* (*reduce_dims_fn)(const mlc_tensor*, const int64_t*,
                                      int64_t, bool);

typedef struct {
    const char* name;
    reduce_fn fn;
    bool is_index; /* argmax/argmin: MLC_I64 result, exact */
    double atol;
    double rtol;
} single_case;

/* Sum and mean: the summation order differs from PyTorch, so the last bits
 * can differ. max/min/median select an input value, so they are exact. */
static const single_case SINGLE[] = {
    {"sum", mlc_sum, false, 1e-5, 1e-5},
    {"prod", mlc_prod, false, 1e-6, 1e-5},
    {"mean", mlc_mean, false, 1e-6, 1e-5},
    {"max", mlc_max, false, 0.0, 0.0},
    {"min", mlc_min, false, 0.0, 0.0},
    {"median", mlc_median, false, 0.0, 0.0},
    {"argmax", mlc_argmax, true, 0.0, 0.0},
    {"argmin", mlc_argmin, true, 0.0, 0.0},
};
#define N_SINGLE (sizeof SINGLE / sizeof SINGLE[0])

typedef struct {
    const char* name;
    reduce_dims_fn fn;
    double atol;
    double rtol;
} multi_case;

static const multi_case MULTI[] = {
    {"sum", mlc_sum_dims, 1e-5, 1e-5},   {"prod", mlc_prod_dims, 1e-6, 1e-5},
    {"mean", mlc_mean_dims, 1e-6, 1e-5}, {"max", mlc_max_dims, 0.0, 0.0},
    {"min", mlc_min_dims, 0.0, 0.0},
};
#define N_MULTI (sizeof MULTI / sizeof MULTI[0])

static int load(golden_file* gf) {
    char err[256] = {0};
    if (golden_load(DATA("ops_reduce.mlct"), gf, err, sizeof err) != 0) {
        mlc_test_fail(__FILE__, __LINE__, "golden_load failed: %s", err);
        return 0;
    }
    return 1;
}

/* Checks a reduction result. `shape` NULL: use the golden shape; otherwise
 * use `shape`/`ndim` (keepdim). */
static int check_result(const mlc_tensor* r, const golden_file* gf,
                        const char* name, bool is_index, const int64_t* shape,
                        int64_t ndim, double atol, double rtol) {
    if (is_index) {
        return tc_check_golden_i64(r, gf, name, shape, ndim);
    }
    if (shape == NULL) {
        return tc_check_golden(r, gf, name, atol, rtol);
    }
    return tc_check_golden_values(r, gf, name, shape, ndim, atol, rtol);
}

/* The keepdim shape: the input shape with every reduced dim set to 1. */
static void keepdim_shape(const mlc_tensor* x, const bool* reduced,
                          int64_t* out) {
    for (int64_t d = 0; d < x->ndim; ++d) {
        out[d] = reduced[d] ? 1 : x->shape[d];
    }
}

/* ------------------------------------------------------------ one dim -- */

/* Every single-dim op over dims 0, 1, 2 and the same dims written as negative
 * numbers, with and without keepdim, on a contiguous [2, 3, 4] input and on a
 * transposed [4, 3, 2] view. Reports every failing case. */
MLC_TEST(single_dim_reductions_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* x = tc_from_golden(&gf, "x");
    mlc_tensor* xt = mlc_transpose(x, 0, 2);
    char k[64];

    for (size_t i = 0; i < N_SINGLE; ++i) {
        const single_case* c = &SINGLE[i];
        for (int64_t dim = 0; dim < 3; ++dim) {
            snprintf(k, sizeof k, "%s_d%lld", c->name, (long long)dim);
            mlc_tensor* r = c->fn(x, dim, false);
            (void)check_result(r, &gf, k, c->is_index, NULL, 0, c->atol,
                               c->rtol);
            mlc_tensor_free(r);

            r = c->fn(x, dim - 3, false); /* the same dim, negative */
            (void)check_result(r, &gf, k, c->is_index, NULL, 0, c->atol,
                               c->rtol);
            mlc_tensor_free(r);

            bool reduced[3] = {false, false, false};
            reduced[dim] = true;
            int64_t kshape[3];
            keepdim_shape(x, reduced, kshape);
            r = c->fn(x, dim, true);
            (void)check_result(r, &gf, k, c->is_index, kshape, 3, c->atol,
                               c->rtol);
            mlc_tensor_free(r);

            snprintf(k, sizeof k, "%s_t_d%lld", c->name, (long long)dim);
            r = c->fn(xt, dim, false);
            (void)check_result(r, &gf, k, c->is_index, NULL, 0, c->atol,
                               c->rtol);
            mlc_tensor_free(r);
        }
    }
    /* No reduction may change its input or keep a reference to it. */
    MLC_ASSERT(tc_check_golden(x, &gf, "x", 0.0, 0.0));
    MLC_ASSERT_EQ_INT(x->data->ref_count, 2); /* x and xt */

    mlc_tensor_free(xt);
    mlc_tensor_free(x);
    golden_free(&gf);
}

/* Ties, NaN, -inf, and +inf, reduced over dim 1 of a [6, 4] input. */
MLC_TEST(single_dim_special_values) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* s = tc_from_golden(&gf, "special");
    char k[64];
    for (size_t i = 0; i < N_SINGLE; ++i) {
        const single_case* c = &SINGLE[i];
        snprintf(k, sizeof k, "%s_special", c->name);
        if (golden_get(&gf, k) == NULL) {
            continue; /* sum/prod/mean: no special case */
        }
        mlc_tensor* r = c->fn(s, 1, false);
        (void)check_result(r, &gf, k, c->is_index, NULL, 0, 0.0, 0.0);
        mlc_tensor_free(r);
    }
    mlc_tensor_free(s);
    golden_free(&gf);
}

/* --------------------------------------------------------- several dims -- */

MLC_TEST(multi_dim_reductions_match_pytorch) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* x = tc_from_golden(&gf, "x");
    struct {
        const char* suffix;
        int64_t dims[3];
        int64_t n;
    } dim_cases[3] = {
        {"d02", {0, 2, 0}, 2}, {"dn1_0", {-1, 0, 0}, 2}, {"all", {0, 1, 2}, 3}};
    char k[64];

    for (size_t i = 0; i < N_MULTI; ++i) {
        const multi_case* c = &MULTI[i];
        for (int j = 0; j < 3; ++j) {
            snprintf(k, sizeof k, "%s_%s", c->name, dim_cases[j].suffix);
            mlc_tensor* r = c->fn(x, dim_cases[j].dims, dim_cases[j].n, false);
            (void)check_result(r, &gf, k, false, NULL, 0, c->atol, c->rtol);
            mlc_tensor_free(r);

            bool reduced[3] = {false, false, false};
            for (int64_t d = 0; d < dim_cases[j].n; ++d) {
                reduced[(dim_cases[j].dims[d] + 3) % 3] = true;
            }
            int64_t kshape[3];
            keepdim_shape(x, reduced, kshape);
            r = c->fn(x, dim_cases[j].dims, dim_cases[j].n, true);
            (void)check_result(r, &gf, k, false, kshape, 3, c->atol, c->rtol);
            mlc_tensor_free(r);
        }
    }
    mlc_tensor_free(x);
    golden_free(&gf);
}

/* The multi-dim API with one dim must give the single-dim result. */
MLC_TEST(multi_dim_api_with_one_dim) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_tensor* x = tc_from_golden(&gf, "x");
    mlc_tensor* r = mlc_sum_dims(x, (int64_t[]){1}, 1, false);
    MLC_ASSERT(tc_check_golden(r, &gf, "sum_d1", 1e-5, 1e-5));
    mlc_tensor_free(r);
    mlc_tensor_free(x);
    golden_free(&gf);
}

/* ---------------------------------------------------------- edge cases -- */

/* Reducing a dim of size 0: sum = 0, prod = 1, mean = NaN (as PyTorch). */
MLC_TEST(reduce_over_empty_dim) {
    mlc_tensor* e = mlc_zeros((int64_t[]){3, 0}, 2, MLC_F32);
    const float zeros[3] = {0, 0, 0};
    const float ones[3] = {1, 1, 1};
    const float nans[3] = {NAN, NAN, NAN};

    mlc_tensor* r = mlc_sum(e, 1, false);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){3}, 1));
    float* v = tc_gather_f32(r);
    MLC_ASSERT_ALLCLOSE_F32(v, zeros, 3, 0.0, 0.0);
    free(v);
    mlc_tensor_free(r);

    r = mlc_prod(e, 1, false);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){3}, 1));
    v = tc_gather_f32(r);
    MLC_ASSERT_ALLCLOSE_F32(v, ones, 3, 0.0, 0.0);
    free(v);
    mlc_tensor_free(r);

    r = mlc_mean(e, 1, false);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){3}, 1));
    v = tc_gather_f32(r);
    MLC_ASSERT_ALLCLOSE_F32(v, nans, 3, 0.0, 0.0);
    free(v);
    mlc_tensor_free(r);
    mlc_tensor_free(e);
}

/* Reducing a non-empty dim of a tensor that has 0 elements: [0, 3] -> [0]. */
MLC_TEST(reduce_tensor_with_zero_rows) {
    mlc_tensor* e = mlc_zeros((int64_t[]){0, 3}, 2, MLC_F32);
    mlc_tensor* r = mlc_sum(e, 1, false);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){0}, 1));
    mlc_tensor_free(r);
    r = mlc_max(e, 1, true);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){0, 1}, 2));
    mlc_tensor_free(r);
    mlc_tensor_free(e);
}

/* 1e6 x 0.1f. The exact sum is 100000.0015; a plain float loop gives about
 * 100958 (1% error). Pairwise or double accumulation is within 1e-5. Checked
 * along a contiguous dim and along a dim with stride 2. */
MLC_TEST(sum_of_one_million_values_is_accurate) {
    const double exact = 100000.00149011612; /* 1e6 * (double)0.1f */
    mlc_tensor* v = mlc_full((int64_t[]){1000000}, 1, MLC_F32, 0.1f);
    mlc_tensor* r = mlc_sum(v, 0, false);
    MLC_ASSERT(tc_check_shape(r, NULL, 0));
    MLC_ASSERT_NEAR(*(const float*)mlc_tensor_data(r), exact, 0.0, 1e-5);
    mlc_tensor_free(r);
    r = mlc_mean(v, 0, false);
    MLC_ASSERT_NEAR(*(const float*)mlc_tensor_data(r), exact / 1e6, 0.0, 1e-5);
    mlc_tensor_free(r);
    mlc_tensor_free(v);

    mlc_tensor* m = mlc_full((int64_t[]){1000000, 2}, 2, MLC_F32, 0.1f);
    r = mlc_sum(m, 0, false);
    MLC_ASSERT(tc_check_shape(r, (int64_t[]){2}, 1));
    float* got = tc_gather_f32(r);
    MLC_ASSERT_NEAR(got[0], exact, 0.0, 1e-5);
    MLC_ASSERT_NEAR(got[1], exact, 0.0, 1e-5);
    free(got);
    mlc_tensor_free(r);
    mlc_tensor_free(m);
}

/* An empty dims list means "reduce nothing" (NumPy's axis=(), decided
 * 2026-10-06): the result is a new contiguous copy with the input's shape and
 * values. PyTorch differs (dim=[] reduces all dims). Checked on inputs that a
 * raw storage copy gets wrong: a slice (offset 3), a transposed view, and an
 * expanded view (6 floats of storage, 6000 elements). */
MLC_TEST(empty_dims_list_is_identity) {
    mlc_tensor* x = mlc_arange(0, 6, 1, MLC_F32);
    mlc_tensor* m = mlc_view(x, (int64_t[]){2, 3}, 2);
    mlc_tensor* col = mlc_view(x, (int64_t[]){6, 1}, 2);
    mlc_tensor* inputs[3] = {mlc_slice(m, 0, 1, 2, 1), mlc_transpose(m, 0, 1),
                             mlc_expand(col, (int64_t[]){6, 1000}, 2)};
    const char* input_names[3] = {"slice", "transpose", "expand"};

    for (int i = 0; i < 3; ++i) {
        mlc_tensor* in = inputs[i];
        const int64_t n = mlc_tensor_numel(in);
        float* expected = tc_gather_f32(in);
        for (size_t j = 0; j < N_MULTI; ++j) {
            for (int keep = 0; keep < 2; ++keep) {
                mlc_tensor* r = MULTI[j].fn(in, NULL, 0, keep == 1);
                if (!tc_check_shape(r, in->shape, in->ndim)) {
                    mlc_test_fail(__FILE__, __LINE__, "%s_dims(%s, []): shape",
                                  MULTI[j].name, input_names[i]);
                    mlc_tensor_free(r);
                    continue;
                }
                if (r->data == in->data || !mlc_is_contiguous(r)) {
                    mlc_test_fail(__FILE__, __LINE__,
                                  "%s_dims(%s, []): result must be a new "
                                  "contiguous tensor",
                                  MULTI[j].name, input_names[i]);
                }
                float* got = tc_gather_f32(r);
                (void)mlc_test_check_allclose_f32(
                    __FILE__, __LINE__, MULTI[j].name, input_names[i], got,
                    expected, (size_t)n, 0.0, 0.0);
                free(got);
                mlc_tensor_free(r);
            }
        }
        free(expected);
    }

    for (int i = 0; i < 3; ++i) {
        mlc_tensor_free(inputs[i]);
    }
    mlc_tensor_free(col);
    mlc_tensor_free(m);
    mlc_tensor_free(x);
}
