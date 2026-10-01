/*
 * test_rng.c - tests for the PRNG (include/mlc/rng.h).
 *
 * Exact outputs are compared with a Python reference (tests/ref/mlinc_ref/
 * rng.py), which tests/ref/gen_rng.py checks against the published splitmix64
 * test vector and the randomgen package. To regenerate the data:
 *     uv run --project tests/ref python tests/ref/gen_rng.py
 *
 * The statistical tests use a fixed seed, so they are deterministic (never
 * flaky). Their tolerances are 5 standard errors of the estimate.
 */
#include <math.h>
#include <stdint.h>

#include "golden.h"
#include "mlc/rng.h"
#include "mlc_test.h"

#define DATA(file) MLC_TEST_DATA_DIR "/" file
#define N_STATS 1000000

/* Compares `n` uint64 values with a golden i64 entry (stored bit patterns). */
static int check_u64(const char* name, const uint64_t* actual,
                     const golden_file* gf, int64_t n) {
    const golden_tensor* g = golden_require(gf, name, GOLDEN_I64);
    if (g->numel != n) {
        mlc_test_fail(__FILE__, __LINE__,
                      "%s: golden has %lld values, not %lld", name,
                      (long long)g->numel, (long long)n);
        return 0;
    }
    const int64_t* expected = (const int64_t*)g->data;
    for (int64_t i = 0; i < n; ++i) {
        /* int64 -> uint64 conversion is defined: it keeps the bit pattern. */
        if (actual[i] != (uint64_t)expected[i]) {
            mlc_test_fail(__FILE__, __LINE__,
                          "%s[%lld]: got 0x%016llx, expected 0x%016llx", name,
                          (long long)i, (unsigned long long)actual[i],
                          (unsigned long long)(uint64_t)expected[i]);
            return 0;
        }
    }
    return 1;
}

static int load(golden_file* gf) {
    char err[256] = {0};
    if (golden_load(DATA("rng.mlct"), gf, err, sizeof err) != 0) {
        mlc_test_fail(__FILE__, __LINE__, "golden_load failed: %s", err);
        return 0;
    }
    return 1;
}

MLC_TEST(splitmix64_reference_vector) {
    /* Direct check of the first published value, without the golden file. */
    uint64_t x = 1234567;
    MLC_ASSERT(mlc_rng_splitmix64(&x) == UINT64_C(6457827717110365317));

    golden_file gf;
    MLC_ASSERT(load(&gf));
    uint64_t out[5];
    x = 1234567;
    for (int i = 0; i < 5; ++i) {
        out[i] = mlc_rng_splitmix64(&x);
    }
    MLC_ASSERT(check_u64("splitmix64_1234567", out, &gf, 5));
    golden_free(&gf);
}

MLC_TEST(xoshiro_from_known_state) {
    /* Well-known xoshiro256** outputs for the state {1, 2, 3, 4}. */
    mlc_rng rng = {{1, 2, 3, 4}};
    MLC_ASSERT(mlc_rng_next(&rng) == 11520);
    MLC_ASSERT(mlc_rng_next(&rng) == 0);
    MLC_ASSERT(mlc_rng_next(&rng) == 1509978240);

    golden_file gf;
    MLC_ASSERT(load(&gf));
    mlc_rng r2 = {{1, 2, 3, 4}};
    uint64_t out[16];
    for (int i = 0; i < 16; ++i) {
        out[i] = mlc_rng_next(&r2);
    }
    MLC_ASSERT(check_u64("next_state_1_2_3_4", out, &gf, 16));
    golden_free(&gf);
}

MLC_TEST(seed_state_and_sequence_match_reference) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    const uint64_t seeds[3] = {0, 1, 42};
    const char* state_names[3] = {"state_seed_0", "state_seed_1",
                                  "state_seed_42"};
    const char* next_names[3] = {"next_seed_0", "next_seed_1", "next_seed_42"};
    for (int s = 0; s < 3; ++s) {
        mlc_rng rng;
        mlc_rng_seed(&rng, seeds[s]);
        MLC_ASSERT(check_u64(state_names[s], rng.state, &gf, 4));
        uint64_t out[16];
        for (int i = 0; i < 16; ++i) {
            out[i] = mlc_rng_next(&rng);
        }
        MLC_ASSERT(check_u64(next_names[s], out, &gf, 16));
    }
    golden_free(&gf);
}

MLC_TEST(same_seed_same_sequence_different_seed_differs) {
    mlc_rng a;
    mlc_rng b;
    mlc_rng c;
    mlc_rng_seed(&a, 123);
    mlc_rng_seed(&b, 123);
    mlc_rng_seed(&c, 124);
    int differs = 0;
    for (int i = 0; i < 100; ++i) {
        const uint64_t va = mlc_rng_next(&a);
        MLC_ASSERT(va == mlc_rng_next(&b));
        differs |= va != mlc_rng_next(&c);
    }
    MLC_ASSERT_MSG(differs, "seeds 123 and 124 give the same sequence");
}

MLC_TEST(randu_matches_reference) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    const golden_tensor* g = golden_require(&gf, "randu_seed_7", GOLDEN_F64);
    double out[64];
    mlc_rng rng;
    mlc_rng_seed(&rng, 7);
    for (int i = 0; i < 64; ++i) {
        out[i] = mlc_rng_randu(&rng);
    }
    /* (x >> 11) * 2^-53 is exact in double, so the match must be exact. */
    MLC_ASSERT_ALLCLOSE_F64(out, (const double*)g->data, 64, 0.0, 0.0);
    golden_free(&gf);
}

MLC_TEST(randu_range_and_mean) {
    mlc_rng rng;
    mlc_rng_seed(&rng, 2024);
    double sum = 0.0;
    double lo = 1.0;
    double hi = 0.0;
    for (int i = 0; i < N_STATS; ++i) {
        const double u = mlc_rng_randu(&rng);
        MLC_ASSERT_MSG(u >= 0.0 && u < 1.0, "sample %d = %.17g not in [0, 1)",
                       i, u);
        sum += u;
        lo = u < lo ? u : lo;
        hi = u > hi ? u : hi;
    }
    /* Uniform [0, 1): mean 0.5, standard error sqrt(1/12 / N). */
    const double mean = sum / N_STATS;
    MLC_ASSERT_NEAR(mean, 0.5, 5.0 * sqrt(1.0 / 12.0 / N_STATS), 0.0);
    MLC_ASSERT_MSG(lo < 1e-4 && hi > 1.0 - 1e-4,
                   "samples do not cover [0, 1): min %g, max %g", lo, hi);
}

MLC_TEST(randn_matches_reference) {
    golden_file gf;
    MLC_ASSERT(load(&gf));
    const golden_tensor* g = golden_require(&gf, "randn_seed_7", GOLDEN_F64);
    double out[64];
    mlc_rng rng;
    mlc_rng_seed(&rng, 7);
    for (int i = 0; i < 64; ++i) {
        out[i] = mlc_rng_randn(&rng);
    }
    /* log, sqrt, and cos can differ in the last bit between C libraries. */
    MLC_ASSERT_ALLCLOSE_F64(out, (const double*)g->data, 64, 1e-15, 1e-12);
    golden_free(&gf);
}

MLC_TEST(randn_mean_variance_and_tails) {
    mlc_rng rng;
    mlc_rng_seed(&rng, 2025);
    double sum = 0.0;
    double sum_sq = 0.0;
    int within_one_sigma = 0;
    for (int i = 0; i < N_STATS; ++i) {
        const double z = mlc_rng_randn(&rng);
        MLC_ASSERT_MSG(isfinite(z), "sample %d is not finite: %g", i, z);
        sum += z;
        sum_sq += z * z;
        within_one_sigma += fabs(z) < 1.0;
    }
    const double mean = sum / N_STATS;
    const double var = sum_sq / N_STATS - mean * mean;
    const double frac = (double)within_one_sigma / N_STATS;
    /* Standard errors: mean 1/sqrt(N), variance sqrt(2/N), and a proportion
     * p = 0.682689 has sqrt(p(1-p)/N). */
    MLC_ASSERT_NEAR(mean, 0.0, 5.0 / sqrt(N_STATS), 0.0);
    MLC_ASSERT_NEAR(var, 1.0, 5.0 * sqrt(2.0 / N_STATS), 0.0);
    MLC_ASSERT_NEAR(frac, 0.682689492,
                    5.0 * sqrt(0.682689 * 0.317311 / N_STATS), 0.0);
}
