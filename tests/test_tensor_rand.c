/*
 * test_tensor_rand.c - tests for mlc_rand and mlc_randn (include/mlc/tensor.h).
 *
 * The statistical tests use a fixed seed, so they are deterministic. Their
 * tolerances are 5 standard errors of the estimate.
 */
#include <math.h>
#include <stdint.h>

#include "mlc/rng.h"
#include "mlc/tensor.h"
#include "mlc_test.h"

/* An xoshiro256** state whose next output is 0xffffffffffffffff, so the next
 * mlc_rng_randu() is the largest value below 1: 1 - 2^-53. The output depends
 * only on s[1]; s[1] = rotr((2^64 - 1) * 9^-1, 7) * 5^-1 (mod 2^64). */
#define MAX_OUTPUT_S1 UINT64_C(0x4fc71c71c71c71c7)

MLC_TEST(rand_shape_and_values_follow_rng) {
    mlc_rng rng;
    mlc_rng ref;
    mlc_rng_seed(&rng, 11);
    mlc_rng_seed(&ref, 11);
    const int64_t shape[2] = {3, 4};
    mlc_tensor* t = mlc_rand(&rng, shape, 2, -2.0f, 3.0f);
    MLC_ASSERT(t != NULL);
    MLC_ASSERT_EQ_INT(t->ndim, 2);
    MLC_ASSERT_EQ_INT(t->shape[0], 3);
    MLC_ASSERT_EQ_INT(t->shape[1], 4);
    MLC_ASSERT(mlc_is_contiguous(t));

    /* Element i uses the i-th draw of the generator. */
    float expected[12];
    for (int i = 0; i < 12; ++i) {
        expected[i] = (float)(-2.0 + 5.0 * mlc_rng_randu(&ref));
    }
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(t), expected, 12,
                            1e-6, 1e-6);
    /* The tensor used exactly 12 draws: both generators are in step. */
    MLC_ASSERT(mlc_rng_next(&rng) == mlc_rng_next(&ref));
    mlc_tensor_free(t);
}

MLC_TEST(rand_same_seed_reproduces_and_draws_advance) {
    mlc_rng a;
    mlc_rng b;
    mlc_rng_seed(&a, 5);
    mlc_rng_seed(&b, 5);
    const int64_t shape[1] = {8};
    mlc_tensor* t1 = mlc_rand(&a, shape, 1, 0.0f, 1.0f);
    mlc_tensor* t2 = mlc_rand(&b, shape, 1, 0.0f, 1.0f);
    mlc_tensor* t3 = mlc_rand(&a, shape, 1, 0.0f, 1.0f);
    MLC_ASSERT(t1 != NULL && t2 != NULL && t3 != NULL);
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(t1),
                            (const float*)mlc_tensor_data(t2), 8, 0.0, 0.0);
    MLC_ASSERT_MSG(((const float*)mlc_tensor_data(t1))[0] !=
                       ((const float*)mlc_tensor_data(t3))[0],
                   "second call gives the same values: rng did not advance");
    mlc_tensor_free(t1);
    mlc_tensor_free(t2);
    mlc_tensor_free(t3);
}

MLC_TEST(rand_range_and_mean) {
    mlc_rng rng;
    mlc_rng_seed(&rng, 99);
    const int64_t shape[2] = {1000, 1000};
    const float lo = -1.5f;
    const float hi = 2.5f;
    mlc_tensor* t = mlc_rand(&rng, shape, 2, lo, hi);
    MLC_ASSERT(t != NULL);
    const float* v = (const float*)mlc_tensor_data(t);
    const int64_t n = mlc_tensor_numel(t);
    double sum = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        MLC_ASSERT_MSG(v[i] >= lo && v[i] < hi,
                       "v[%lld] = %.9g not in [%g, %g)", (long long)i, v[i], lo,
                       hi);
        sum += v[i];
    }
    /* Uniform [lo, hi): mean (lo+hi)/2, standard error (hi-lo)/sqrt(12 N). */
    MLC_ASSERT_NEAR(sum / (double)n, 0.5 * (lo + hi),
                    5.0 * (hi - lo) / sqrt(12.0 * (double)n), 0.0);
    mlc_tensor_free(t);
}

/* The largest draw (1 - 2^-53) must still give a value below `max`.
 * (float)(1 - 2^-53) rounds to 1.0f, so min + (max - min) * u can equal max. */
MLC_TEST(rand_never_returns_max) {
    mlc_rng rng = {{0, MAX_OUTPUT_S1, 0, 0}};
    mlc_rng probe = rng;
    MLC_ASSERT_MSG(mlc_rng_next(&probe) == UINT64_MAX,
                   "test setup: crafted state does not give the max output");

    /* Positive, negative, zero, and large `max`. A fix that steps "toward 0"
     * works only for the first one. Above 2^24, floats are more than 1 apart,
     * so a fix that steps "toward max - 1" fails for the last one. */
    const float ranges[4][2] = {
        {1.0f, 2.0f}, {-3.0f, -1.0f}, {-1.0f, 0.0f}, {1e8f, 2e8f}};
    for (int r = 0; r < 4; ++r) {
        const float lo = ranges[r][0];
        const float hi = ranges[r][1];
        mlc_rng crafted = {{0, MAX_OUTPUT_S1, 0, 0}};
        mlc_tensor* t = mlc_rand(&crafted, NULL, 0, lo, hi);
        MLC_ASSERT(t != NULL);
        const float v = *(const float*)mlc_tensor_data(t);
        mlc_tensor_free(t);
        MLC_ASSERT_MSG(v >= lo && v < hi,
                       "rand(%g, %g) returned %.9g for the largest draw; the "
                       "range is [min, max), so the value must be < max",
                       lo, hi, v);
    }
}

MLC_TEST(rand_zero_size_does_not_draw) {
    mlc_rng rng;
    mlc_rng ref;
    mlc_rng_seed(&rng, 3);
    mlc_rng_seed(&ref, 3);
    const int64_t shape[2] = {0, 5};
    mlc_tensor* t = mlc_rand(&rng, shape, 2, 0.0f, 1.0f);
    MLC_ASSERT(t != NULL);
    MLC_ASSERT_EQ_INT(mlc_tensor_numel(t), 0);
    MLC_ASSERT(mlc_rng_next(&rng) == mlc_rng_next(&ref));
    mlc_tensor_free(t);
}

MLC_TEST(randn_values_follow_rng) {
    mlc_rng rng;
    mlc_rng ref;
    mlc_rng_seed(&rng, 17);
    mlc_rng_seed(&ref, 17);
    const int64_t shape[1] = {16};
    mlc_tensor* t = mlc_randn(&rng, shape, 1, 0.5f, 2.0f);
    MLC_ASSERT(t != NULL);
    float expected[16];
    for (int i = 0; i < 16; ++i) {
        expected[i] = (float)(0.5 + 2.0 * mlc_rng_randn(&ref));
    }
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(t), expected, 16,
                            1e-6, 1e-6);
    mlc_tensor_free(t);
}

MLC_TEST(randn_mean_and_std) {
    mlc_rng rng;
    mlc_rng_seed(&rng, 1234);
    const int64_t shape[2] = {1000, 1000};
    const double mu = -3.0;
    const double sigma = 0.25;
    mlc_tensor* t = mlc_randn(&rng, shape, 2, (float)mu, (float)sigma);
    MLC_ASSERT(t != NULL);
    const float* v = (const float*)mlc_tensor_data(t);
    const int64_t n = mlc_tensor_numel(t);
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        MLC_ASSERT_MSG(isfinite(v[i]), "v[%lld] is not finite", (long long)i);
        sum += v[i];
        sum_sq += (double)v[i] * v[i];
    }
    const double mean = sum / (double)n;
    const double std = sqrt(sum_sq / (double)n - mean * mean);
    /* Standard errors: mean sigma/sqrt(N), std sigma/sqrt(2N). */
    MLC_ASSERT_NEAR(mean, mu, 5.0 * sigma / sqrt((double)n), 0.0);
    MLC_ASSERT_NEAR(std, sigma, 5.0 * sigma / sqrt(2.0 * (double)n), 0.0);
    mlc_tensor_free(t);
}
