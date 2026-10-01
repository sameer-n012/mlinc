/*
 * test_tensor_view.c - tests for views, mlc_contiguous, mlc_reshape, and
 * mlc_clone (include/mlc/tensor.h). This is the Phase 1 milestone test.
 *
 * The test reads tensor values with its own strided gather (gather_f32), so
 * the view tests do not depend on mlc_contiguous being correct.
 *
 * Golden data: tests/ref/gen_tensor_views.py. To regenerate it:
 *     uv run --project tests/ref python tests/ref/gen_tensor_views.py
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "golden.h"
#include "mlc/tensor.h"
#include "mlc_test.h"

#define DATA(file) MLC_TEST_DATA_DIR "/" file

/* Returns the element of `t` at the multi-index `idx`, using only the shape,
 * strides, and offset fields. */
static float at(const mlc_tensor* t, const int64_t* idx) {
    int64_t pos = t->offset;
    for (int64_t d = 0; d < t->ndim; ++d) {
        pos += idx[d] * t->strides[d];
    }
    return ((const float*)t->data->data)[pos];
}

/* Copies all elements of `t` in row-major order into a new malloc'd array.
 * The caller frees it. Returns NULL if numel is 0 or malloc fails. */
static float* gather_f32(const mlc_tensor* t) {
    const int64_t n = mlc_tensor_numel(t);
    float* out = n > 0 ? malloc((size_t)n * sizeof(float)) : NULL;
    if (out == NULL) {
        return NULL;
    }
    int64_t idx[MLC_MAX_DIMS] = {0};
    for (int64_t i = 0; i < n; ++i) {
        out[i] = at(t, idx);
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

/* Compares the shape of `t` with `shape`. */
static int shape_is(const mlc_tensor* t, const int64_t* shape, int64_t ndim) {
    if (t == NULL || t->ndim != ndim) {
        mlc_test_fail(__FILE__, __LINE__, "ndim: got %lld, expected %lld",
                      t != NULL ? (long long)t->ndim : -1LL, (long long)ndim);
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

/* Compares shape and values of `t` with a golden entry. */
static int matches_golden(const mlc_tensor* t, const golden_tensor* g) {
    if (!shape_is(t, g->shape, (int64_t)g->ndim)) {
        mlc_test_fail(__FILE__, __LINE__, "case '%s': shape differs", g->name);
        return 0;
    }
    float* values = gather_f32(t);
    const int ok = mlc_test_check_allclose_f32(
        __FILE__, __LINE__, g->name, "golden", values, (const float*)g->data,
        (size_t)g->numel, 0.0, 0.0);
    free(values);
    return ok;
}

/* base = arange(24).view(2, 3, 4), the input of every golden case. */
static mlc_tensor* make_base(void) {
    mlc_tensor* flat = mlc_arange(0, 24, 1, MLC_F32);
    mlc_tensor* base = mlc_view(flat, (int64_t[]){2, 3, 4}, 3);
    mlc_tensor_free(flat);
    return base;
}

MLC_TEST(view_shares_storage_and_writes_through) {
    mlc_tensor* flat = mlc_arange(0, 6, 1, MLC_F32);
    mlc_tensor* v = mlc_view(flat, (int64_t[]){2, 3}, 2);
    MLC_ASSERT(shape_is(v, (int64_t[]){2, 3}, 2));
    MLC_ASSERT(v->data == flat->data);
    MLC_ASSERT_EQ_INT(flat->data->ref_count, 2);
    MLC_ASSERT_EQ_INT(v->strides[0], 3);
    MLC_ASSERT_EQ_INT(v->strides[1], 1);

    /* v[1][2] is flat[5]. */
    ((float*)mlc_tensor_data(v))[1 * 3 + 2] = 42.0f;
    MLC_ASSERT(((const float*)mlc_tensor_data(flat))[5] == 42.0f);

    mlc_tensor_free(v);
    MLC_ASSERT_EQ_INT(flat->data->ref_count, 1);
    mlc_tensor_free(flat);
}

MLC_TEST(view_keeps_storage_alive_after_base_is_freed) {
    mlc_tensor* flat = mlc_arange(0, 6, 1, MLC_F32);
    mlc_tensor* t = mlc_transpose(flat, 0, 0);
    mlc_tensor_free(flat);
    /* ASan reports a use-after-free here if the storage was freed. */
    MLC_ASSERT_EQ_INT(t->data->ref_count, 1);
    MLC_ASSERT(at(t, (int64_t[]){4}) == 4.0f);
    mlc_tensor_free(t);
}

MLC_TEST(transpose_strides_and_contiguity) {
    mlc_tensor* flat = mlc_arange(0, 6, 1, MLC_F32);
    mlc_tensor* m = mlc_view(flat, (int64_t[]){2, 3}, 2);
    mlc_tensor* t = mlc_transpose(m, 0, 1);
    MLC_ASSERT(shape_is(t, (int64_t[]){3, 2}, 2));
    MLC_ASSERT_EQ_INT(t->strides[0], 1);
    MLC_ASSERT_EQ_INT(t->strides[1], 3);
    MLC_ASSERT(!mlc_is_contiguous(t));

    mlc_tensor* c = mlc_contiguous(t);
    MLC_ASSERT(mlc_is_contiguous(c));
    MLC_ASSERT(c->data != m->data);
    const float expected[6] = {0, 3, 1, 4, 2, 5};
    MLC_ASSERT_ALLCLOSE_F32((const float*)mlc_tensor_data(c), expected, 6, 0.0,
                            0.0);
    mlc_tensor_free(c);
    mlc_tensor_free(t);
    mlc_tensor_free(m);
    mlc_tensor_free(flat);
}

MLC_TEST(slice_values_offset_and_step) {
    mlc_tensor* t = mlc_arange(0, 10, 1, MLC_F32);

    mlc_tensor* s = mlc_slice(t, 0, 1, 8, 3); /* 1, 4, 7 */
    MLC_ASSERT(shape_is(s, (int64_t[]){3}, 1));
    MLC_ASSERT_EQ_INT(s->offset, 1);
    MLC_ASSERT_EQ_INT(s->strides[0], 3);
    MLC_ASSERT(*(const float*)mlc_tensor_data(s) == 1.0f);
    float* v = gather_f32(s);
    MLC_ASSERT_ALLCLOSE_F32(v, ((const float[]){1, 4, 7}), 3, 0.0, 0.0);
    free(v);
    mlc_tensor_free(s);

    s = mlc_slice(t, -1, -3, 10, 1); /* negative dim and start: 7, 8, 9 */
    v = gather_f32(s);
    MLC_ASSERT(shape_is(s, (int64_t[]){3}, 1));
    MLC_ASSERT_ALLCLOSE_F32(v, ((const float[]){7, 8, 9}), 3, 0.0, 0.0);
    free(v);
    mlc_tensor_free(s);

    s = mlc_slice(t, 0, 4, 4, 1); /* empty */
    MLC_ASSERT(shape_is(s, (int64_t[]){0}, 1));
    mlc_tensor_free(s);

    mlc_tensor_free(t);
}

/* A view of a sliced tensor must keep the slice's offset. */
MLC_TEST(views_of_a_slice_keep_offset) {
    mlc_tensor* flat = mlc_arange(0, 12, 1, MLC_F32);
    mlc_tensor* m = mlc_view(flat, (int64_t[]){4, 3}, 2);
    mlc_tensor* rows = mlc_slice(m, 0, 1, 3, 1); /* values 3..8 */
    MLC_ASSERT(mlc_is_contiguous(rows));

    mlc_tensor* v = mlc_view(rows, (int64_t[]){6}, 1);
    MLC_ASSERT_MSG(v->offset == rows->offset,
                   "mlc_view: offset %lld, expected %lld", (long long)v->offset,
                   (long long)rows->offset);
    MLC_ASSERT(*(const float*)mlc_tensor_data(v) == 3.0f);

    mlc_tensor* t = mlc_transpose(rows, 0, 1);
    MLC_ASSERT_MSG(t->offset == rows->offset,
                   "mlc_transpose: offset %lld, expected %lld",
                   (long long)t->offset, (long long)rows->offset);
    MLC_ASSERT(at(t, (int64_t[]){2, 1}) == 8.0f);

    mlc_tensor* u = mlc_unsqueeze(rows, 0);
    MLC_ASSERT_MSG(u->offset == rows->offset,
                   "mlc_unsqueeze: offset %lld, expected %lld",
                   (long long)u->offset, (long long)rows->offset);

    mlc_tensor_free(u);
    mlc_tensor_free(t);
    mlc_tensor_free(v);
    mlc_tensor_free(rows);
    mlc_tensor_free(m);
    mlc_tensor_free(flat);
}

MLC_TEST(expand_uses_stride_zero) {
    mlc_tensor* flat = mlc_arange(0, 3, 1, MLC_F32);
    mlc_tensor* col = mlc_view(flat, (int64_t[]){3, 1}, 2);
    mlc_tensor* e = mlc_expand(col, (int64_t[]){2, 3, 4}, 3);
    MLC_ASSERT(shape_is(e, (int64_t[]){2, 3, 4}, 3));
    MLC_ASSERT_EQ_INT(e->strides[0], 0);
    MLC_ASSERT_EQ_INT(e->strides[1], 1);
    MLC_ASSERT_EQ_INT(e->strides[2], 0);
    MLC_ASSERT(!mlc_is_contiguous(e));
    MLC_ASSERT(e->data == flat->data);
    mlc_tensor_free(e);
    mlc_tensor_free(col);
    mlc_tensor_free(flat);
}

MLC_TEST(squeeze_and_unsqueeze_shapes) {
    mlc_tensor* base = make_base();
    mlc_tensor* u0 = mlc_unsqueeze(base, 0);
    MLC_ASSERT(shape_is(u0, (int64_t[]){1, 2, 3, 4}, 4));
    MLC_ASSERT(mlc_is_contiguous(u0));
    mlc_tensor* u_last = mlc_unsqueeze(base, -1);
    MLC_ASSERT(shape_is(u_last, (int64_t[]){2, 3, 4, 1}, 4));
    mlc_tensor* s = mlc_squeeze(u0, 0);
    MLC_ASSERT(shape_is(s, (int64_t[]){2, 3, 4}, 3));
    /* An unsqueezed contiguous tensor can be viewed again. */
    mlc_tensor* v = mlc_view(u_last, (int64_t[]){24}, 1);
    MLC_ASSERT(shape_is(v, (int64_t[]){24}, 1));
    mlc_tensor_free(v);
    mlc_tensor_free(s);
    mlc_tensor_free(u_last);
    mlc_tensor_free(u0);
    mlc_tensor_free(base);
}

MLC_TEST(reshape_views_when_contiguous_and_copies_otherwise) {
    mlc_tensor* base = make_base();
    mlc_tensor* r = mlc_reshape(base, (int64_t[]){6, 4}, 2);
    MLC_ASSERT(r->data == base->data); /* contiguous input: a view */

    mlc_tensor* t = mlc_transpose(base, 0, 2);
    mlc_tensor* rt = mlc_reshape(t, (int64_t[]){6, 4}, 2);
    MLC_ASSERT(rt->data != base->data); /* non-contiguous input: a copy */
    MLC_ASSERT(mlc_is_contiguous(rt));
    MLC_ASSERT_EQ_INT(rt->data->ref_count, 1);

    mlc_tensor_free(rt);
    mlc_tensor_free(t);
    mlc_tensor_free(r);
    mlc_tensor_free(base);
}

MLC_TEST(clone_is_independent) {
    mlc_tensor* base = make_base();
    mlc_tensor* c = mlc_clone(base);
    MLC_ASSERT(c->data != base->data);
    ((float*)mlc_tensor_data(c))[0] = -1.0f;
    MLC_ASSERT(((const float*)mlc_tensor_data(base))[0] == 0.0f);
    mlc_tensor_free(c);
    mlc_tensor_free(base);
}

/* Builds the view chain for one golden case. Must match gen_tensor_views.py.
 * Returns the result; frees all temporary tensors. */
static mlc_tensor* build_case(const char* name, mlc_tensor* base) {
    if (strcmp(name, "permute_2_0_1") == 0) {
        return mlc_permute(base, (int64_t[]){2, 0, 1});
    }
    if (strcmp(name, "transpose_0_2") == 0) {
        return mlc_transpose(base, 0, 2);
    }
    if (strcmp(name, "transpose_neg") == 0) {
        return mlc_transpose(base, -1, -2);
    }
    if (strcmp(name, "slice_d1_1_3") == 0) {
        return mlc_slice(base, 1, 1, 3, 1);
    }
    if (strcmp(name, "slice_d2_step2") == 0) {
        return mlc_slice(base, 2, 0, 4, 2);
    }
    if (strcmp(name, "slice_neg_start") == 0) {
        return mlc_slice(base, 2, -3, 4, 1);
    }
    if (strcmp(name, "slice_then_transpose") == 0) {
        mlc_tensor* s = mlc_slice(base, 1, 1, 3, 1);
        mlc_tensor* r = mlc_transpose(s, -1, -2);
        mlc_tensor_free(s);
        return r;
    }
    if (strcmp(name, "slice_then_view") == 0) {
        mlc_tensor* s = mlc_slice(base, 0, 1, 2, 1);
        mlc_tensor* r = mlc_reshape(s, (int64_t[]){12}, 1);
        mlc_tensor_free(s);
        return r;
    }
    if (strcmp(name, "expand_3x1_to_2x3x4") == 0) {
        mlc_tensor* flat = mlc_arange(0, 3, 1, MLC_F32);
        mlc_tensor* col = mlc_view(flat, (int64_t[]){3, 1}, 2);
        mlc_tensor* r = mlc_expand(col, (int64_t[]){2, 3, 4}, 3);
        mlc_tensor_free(col);
        mlc_tensor_free(flat);
        return r;
    }
    if (strcmp(name, "unsqueeze_then_permute") == 0) {
        mlc_tensor* u = mlc_unsqueeze(base, 1);
        mlc_tensor* r = mlc_permute(u, (int64_t[]){3, 1, 0, 2});
        mlc_tensor_free(u);
        return r;
    }
    if (strcmp(name, "squeeze_after_slice") == 0) {
        mlc_tensor* s = mlc_slice(base, 1, 1, 2, 1);
        mlc_tensor* r = mlc_squeeze(s, 1);
        mlc_tensor_free(s);
        return r;
    }
    if (strcmp(name, "reshape_of_transpose") == 0) {
        mlc_tensor* t = mlc_transpose(base, 0, 2);
        mlc_tensor* r = mlc_reshape(t, (int64_t[]){6, 4}, 2);
        mlc_tensor_free(t);
        return r;
    }
    return NULL;
}

/* Runs every golden case, and reports all failing cases, not only the first. */
MLC_TEST(views_match_pytorch) {
    golden_file gf;
    char err[256] = {0};
    MLC_ASSERT_MSG(
        golden_load(DATA("tensor_views.mlct"), &gf, err, sizeof err) == 0,
        "golden_load failed: %s", err);
    mlc_tensor* base = make_base();

    for (uint32_t i = 0; i < gf.count; ++i) {
        const golden_tensor* g = &gf.tensors[i];
        mlc_tensor* r = build_case(g->name, base);
        if (r == NULL) {
            mlc_test_fail(__FILE__, __LINE__, "no C code for case '%s'",
                          g->name);
            continue;
        }
        (void)matches_golden(r, g);
        /* The view result must also match after mlc_contiguous. */
        mlc_tensor* c = mlc_contiguous(r);
        if (!mlc_is_contiguous(c)) {
            mlc_test_fail(__FILE__, __LINE__,
                          "case '%s': mlc_contiguous result is not contiguous",
                          g->name);
        }
        (void)matches_golden(c, g);
        mlc_tensor_free(c);
        mlc_tensor_free(r);
    }

    mlc_tensor_free(base);
    golden_free(&gf);
}
