/*
 * mlc_test.c - test registry, assertion helpers, and main() for the mlinc test
 * harness. See mlc_test.h for the usage.
 */
#include "mlc_test.h"

#include <inttypes.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MLC_TEST_MAX 512

typedef struct {
    const char* name;
    mlc_test_fn fn;
} mlc_test_entry;

static mlc_test_entry g_tests[MLC_TEST_MAX];
static size_t g_test_count = 0;
/* Set by mlc_test_fail; main() resets it before each test. */
static int g_current_failed = 0;

void mlc_test_register(const char* name, mlc_test_fn fn) {
    /* This runs before main(), so abort is the only safe way to report the
     * error. */
    if (g_test_count == MLC_TEST_MAX) {
        fprintf(stderr, "mlc_test: more than %d tests in one binary\n",
                MLC_TEST_MAX);
        abort();
    }
    g_tests[g_test_count].name = name;
    g_tests[g_test_count].fn = fn;
    g_test_count++;
}

void mlc_test_fail(const char* file, int line, const char* fmt, ...) {
    g_current_failed = 1;
    /* All output goes to stdout, so failure details and test status stay in
     * order. */
    printf("  %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    putchar('\n');
}

int mlc_test_check_eq_int(const char* file, int line, const char* expr_a,
                          const char* expr_b, int64_t a, int64_t b) {
    if (a == b) {
        return 1;
    }
    mlc_test_fail(file, line, "%s == %s failed: %" PRId64 " vs %" PRId64,
                  expr_a, expr_b, a, b);
    return 0;
}

/* numpy.isclose semantics, with equal_nan=True. */
static int is_close(double actual, double expected, double atol, double rtol) {
    if (isnan(actual) || isnan(expected)) {
        return isnan(actual) && isnan(expected);
    }
    if (isinf(actual) || isinf(expected)) {
        return actual == expected;
    }
    return fabs(actual - expected) <= atol + rtol * fabs(expected);
}

int mlc_test_check_near(const char* file, int line, const char* expr_a,
                        const char* expr_b, double actual, double expected,
                        double atol, double rtol) {
    if (is_close(actual, expected, atol, rtol)) {
        return 1;
    }
    mlc_test_fail(
        file, line,
        "%s ~= %s failed: %.9g vs %.9g (|diff| %.3g, atol %.3g, rtol %.3g)",
        expr_a, expr_b, actual, expected, fabs(actual - expected), atol, rtol);
    return 0;
}

/* Shared body of the f32 and f64 allclose checks. `load` reads element i as a
 * double. */
static int check_allclose(const char* file, int line, const char* expr_a,
                          const char* expr_b, const void* actual,
                          const void* expected, size_t n,
                          double (*load)(const void*, size_t), double atol,
                          double rtol) {
    if (n > 0 && (actual == NULL || expected == NULL)) {
        mlc_test_fail(file, line, "%s vs %s: NULL array", expr_a, expr_b);
        return 0;
    }
    size_t mismatches = 0;
    size_t first_bad = 0;
    size_t worst = 0;
    double worst_diff = -1.0;
    for (size_t i = 0; i < n; i++) {
        const double a = load(actual, i);
        const double e = load(expected, i);
        if (is_close(a, e, atol, rtol)) {
            continue;
        }
        if (mismatches == 0) {
            first_bad = i;
        }
        mismatches++;
        /* A NaN or inf mismatch has no finite difference; report it as
         * infinite. */
        const double diff =
            (isfinite(a) && isfinite(e)) ? fabs(a - e) : INFINITY;
        if (diff > worst_diff) {
            worst_diff = diff;
            worst = i;
        }
    }
    if (mismatches == 0) {
        return 1;
    }
    mlc_test_fail(
        file, line,
        "%s vs %s: %zu of %zu elements differ (atol %.3g, rtol %.3g)\n"
        "    first at [%zu]: %.9g vs %.9g\n"
        "    worst at [%zu]: %.9g vs %.9g (|diff| %.3g)",
        expr_a, expr_b, mismatches, n, atol, rtol, first_bad,
        load(actual, first_bad), load(expected, first_bad), worst,
        load(actual, worst), load(expected, worst), worst_diff);
    return 0;
}

static double load_f32(const void* p, size_t i) {
    return (double)((const float*)p)[i];
}

static double load_f64(const void* p, size_t i) {
    return ((const double*)p)[i];
}

int mlc_test_check_allclose_f32(const char* file, int line, const char* expr_a,
                                const char* expr_b, const float* actual,
                                const float* expected, size_t n, double atol,
                                double rtol) {
    return check_allclose(file, line, expr_a, expr_b, actual, expected, n,
                          load_f32, atol, rtol);
}

int mlc_test_check_allclose_f64(const char* file, int line, const char* expr_a,
                                const char* expr_b, const double* actual,
                                const double* expected, size_t n, double atol,
                                double rtol) {
    return check_allclose(file, line, expr_a, expr_b, actual, expected, n,
                          load_f64, atol, rtol);
}

static const char* base_name(const char* path) {
    const char* slash = strrchr(path, '/');
    return slash != NULL ? slash + 1 : path;
}

int main(int argc, char** argv) {
    const char* prog = base_name(argc > 0 ? argv[0] : "test");
    const char* filter = NULL;

    if (argc > 2) {
        fprintf(stderr, "usage: %s [--list | <name substring>]\n", prog);
        return 2;
    }
    if (argc == 2) {
        if (strcmp(argv[1], "--list") == 0) {
            for (size_t i = 0; i < g_test_count; i++) {
                printf("%s\n", g_tests[i].name);
            }
            return 0;
        }
        filter = argv[1];
    }

    size_t run = 0;
    size_t failed = 0;
    for (size_t i = 0; i < g_test_count; i++) {
        if (filter != NULL && strstr(g_tests[i].name, filter) == NULL) {
            continue;
        }
        printf("[ RUN  ] %s\n", g_tests[i].name);
        /* Flush before the test runs: if the test crashes, the output shows
         * which test. */
        fflush(stdout);
        g_current_failed = 0;
        g_tests[i].fn();
        run++;
        if (g_current_failed) {
            failed++;
            printf("[ FAIL ] %s\n", g_tests[i].name);
        } else {
            printf("[  OK  ] %s\n", g_tests[i].name);
        }
    }

    if (run == 0) {
        /* A binary that runs no tests is almost always a mistake (bad filter,
         * no MLC_TEST). */
        printf("%s: no tests ran%s%s\n", prog,
               filter != NULL ? " for filter " : "",
               filter != NULL ? filter : "");
        return 1;
    }
    printf("%s: %zu passed, %zu failed\n", prog, run - failed, failed);
    return failed == 0 ? 0 : 1;
}
