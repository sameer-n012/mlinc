#ifndef MLC_KERNEL_H
#define MLC_KERNEL_H

#include "stdint.h"
#include "mlc/tensor.h"

// Row-wise function pointer types for unary and binary operations on tensors
typedef void (*mlc_binary_row_fn)(const float* a, int64_t sa, const float* b,
    int64_t sb, float* out, int64_t so, int64_t n);
typedef void (*mlc_unary_row_fn)(const float* a, int64_t sa, float* out,
    int64_t so, int64_t n);

// Defined in ops_unary.c and ops_binary.c
mlc_tensor* mlc_apply_unary(const mlc_tensor* a, mlc_unary_row_fn fn);
mlc_tensor* mlc_apply_unary_inplace(mlc_tensor* a, mlc_unary_row_fn fn);
mlc_tensor* mlc_apply_binary(const mlc_tensor* a, const mlc_tensor* b, mlc_binary_row_fn fn);
mlc_tensor* mlc_apply_binary_inplace(mlc_tensor* a, const mlc_tensor* b, mlc_binary_row_fn fn);

bool mlc_check_fastpath_unary(const mlc_tensor* a);
bool mlc_check_fastpath_binary(const mlc_tensor* a, const mlc_tensor* b);

#endif
