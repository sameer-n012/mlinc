/*
 * mlc_test.h - minimal test harness for mlinc.
 *
 * Each tests/test_*.c file is one test binary. The file defines tests with
 * MLC_TEST and has no main(): tests/support/mlc_test.c supplies main().
 *
 *     #include "mlc_test.h"
 *
 *     MLC_TEST(add_two_numbers) {
 *         MLC_ASSERT_EQ_INT(1 + 1, 2);
 *         MLC_ASSERT_NEAR(0.1 + 0.2, 0.3, 1e-12, 0.0);
 *     }
 *
 * Command line:
 *     test_foo            run all tests in the binary
 *     test_foo <substr>   run only the tests whose name contains <substr>
 *     test_foo --list     print the test names
 * The exit code is 0 if all tests that ran passed, and 1 otherwise.
 *
 * Rules:
 * - Tests register automatically at start-up. The run order is NOT
 *   guaranteed, so a test must not depend on another test.
 * - A failed MLC_ASSERT* returns from the test function at once. Free
 *   resources in a way that is safe after an early return, or accept a leak
 *   in a failed test.
 * - Tolerances follow numpy.isclose:
 *       |actual - expected| <= atol + rtol * |expected|
 *   NaN matches only NaN; +inf and -inf match only the same infinity.
 */
#ifndef MLC_TEST_H
#define MLC_TEST_H

#include <stddef.h>
#include <stdint.h>

#if defined(__GNUC__) || defined(__clang__)
#define MLC_TEST_CONSTRUCTOR __attribute__((constructor))
#define MLC_TEST_PRINTF(fmt_idx, arg_idx) \
    __attribute__((format(printf, fmt_idx, arg_idx)))
#else
#error "mlc_test.h needs GCC or Clang (it uses __attribute__((constructor)))"
#endif

typedef void (*mlc_test_fn)(void);

/* Internal functions. Use the macros below. */
void mlc_test_register(const char* name, mlc_test_fn fn);
void mlc_test_fail(const char* file, int line, const char* fmt, ...)
    MLC_TEST_PRINTF(3, 4);
int mlc_test_check_eq_int(const char* file, int line, const char* expr_a,
                          const char* expr_b, int64_t a, int64_t b);
int mlc_test_check_near(const char* file, int line, const char* expr_a,
                        const char* expr_b, double actual, double expected,
                        double atol, double rtol);
int mlc_test_check_allclose_f32(const char* file, int line, const char* expr_a,
                                const char* expr_b, const float* actual,
                                const float* expected, size_t n, double atol,
                                double rtol);
int mlc_test_check_allclose_f64(const char* file, int line, const char* expr_a,
                                const char* expr_b, const double* actual,
                                const double* expected, size_t n, double atol,
                                double rtol);

/* Defines and registers a test. `name` must be a valid C identifier, unique in
 * the file. */
#define MLC_TEST(name)                                           \
    static void mlc_test_fn_##name(void);                        \
    MLC_TEST_CONSTRUCTOR static void mlc_test_reg_##name(void) { \
        mlc_test_register(#name, mlc_test_fn_##name);            \
    }                                                            \
    static void mlc_test_fn_##name(void)

/* Fails the test if `cond` is false. */
#define MLC_ASSERT(cond)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            mlc_test_fail(__FILE__, __LINE__, "assertion failed: %s", #cond); \
            return;                                                           \
        }                                                                     \
    } while (0)

/* Fails the test with a printf-style message if `cond` is false. The format
 * string is the first variadic argument, so the macro is valid C11 without the
 * GNU ## extension. */
#define MLC_ASSERT_MSG(cond, ...)                           \
    do {                                                    \
        if (!(cond)) {                                      \
            mlc_test_fail(__FILE__, __LINE__, __VA_ARGS__); \
            return;                                         \
        }                                                   \
    } while (0)

/* Integer equality. Both values are converted to int64_t. */
#define MLC_ASSERT_EQ_INT(actual, expected)                                   \
    do {                                                                      \
        if (!mlc_test_check_eq_int(__FILE__, __LINE__, #actual, #expected,    \
                                   (int64_t)(actual), (int64_t)(expected))) { \
            return;                                                           \
        }                                                                     \
    } while (0)

/* Scalar floating-point comparison with an absolute and a relative tolerance.
 */
#define MLC_ASSERT_NEAR(actual, expected, atol, rtol)                          \
    do {                                                                       \
        if (!mlc_test_check_near(__FILE__, __LINE__, #actual, #expected,       \
                                 (double)(actual), (double)(expected), (atol), \
                                 (rtol))) {                                    \
            return;                                                            \
        }                                                                      \
    } while (0)

/* Element-wise comparison of two contiguous arrays of `n` elements. On failure,
 * prints the number of mismatches, the first mismatch, and the largest absolute
 * difference. */
#define MLC_ASSERT_ALLCLOSE_F32(actual, expected, n, atol, rtol)               \
    do {                                                                       \
        if (!mlc_test_check_allclose_f32(__FILE__, __LINE__, #actual,          \
                                         #expected, (actual), (expected), (n), \
                                         (atol), (rtol))) {                    \
            return;                                                            \
        }                                                                      \
    } while (0)

#define MLC_ASSERT_ALLCLOSE_F64(actual, expected, n, atol, rtol)               \
    do {                                                                       \
        if (!mlc_test_check_allclose_f64(__FILE__, __LINE__, #actual,          \
                                         #expected, (actual), (expected), (n), \
                                         (atol), (rtol))) {                    \
            return;                                                            \
        }                                                                      \
    } while (0)

#endif /* MLC_TEST_H */
