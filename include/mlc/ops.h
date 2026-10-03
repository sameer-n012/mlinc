#ifndef MLC_OPS_H
#define MLC_OPS_H

#include "mlc/tensor.h"
#include "mlc/error.h"

// Unary Ops
mlc_tensor* mlc_neg(const mlc_tensor* tensor);
mlc_tensor* mlc_abs(const mlc_tensor* tensor);
mlc_tensor* mlc_exp(const mlc_tensor* tensor);
mlc_tensor* mlc_log(const mlc_tensor* tensor);
mlc_tensor* mlc_sqrt(const mlc_tensor* tensor);
mlc_tensor* mlc_tanh(const mlc_tensor* tensor);
mlc_tensor* mlc_sigmoid(const mlc_tensor* tensor);
mlc_tensor* mlc_relu(const mlc_tensor* tensor);
mlc_tensor* mlc_gelu(const mlc_tensor* tensor);

// Unary Inplace Ops
mlc_status mlc_neg_(mlc_tensor* tensor);
mlc_status mlc_abs_(mlc_tensor* tensor);
mlc_status mlc_exp_(mlc_tensor* tensor);
mlc_status mlc_log_(mlc_tensor* tensor);
mlc_status mlc_sqrt_(mlc_tensor* tensor);
mlc_status mlc_tanh_(mlc_tensor* tensor);
mlc_status mlc_sigmoid_(mlc_tensor* tensor);
mlc_status mlc_relu_(mlc_tensor* tensor);
mlc_status mlc_gelu_(mlc_tensor* tensor);

// Binary Ops
mlc_tensor* mlc_add(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_sub(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_mul(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_div(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_pow(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_maximum(const mlc_tensor* a, const mlc_tensor* b);
mlc_tensor* mlc_minimum(const mlc_tensor* a, const mlc_tensor* b);

// Binary Inplace Ops
mlc_status mlc_add_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_sub_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_mul_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_div_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_pow_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_maximum_(mlc_tensor* a, const mlc_tensor* b);
mlc_status mlc_minimum_(mlc_tensor* a, const mlc_tensor* b);

// Reduction Ops (Single dimension)
mlc_tensor* mlc_sum(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_prod(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_mean(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_median(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_max(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_min(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_argmax(const mlc_tensor* a, int64_t dim, bool keepdim);
mlc_tensor* mlc_argmin(const mlc_tensor* a, int64_t dim, bool keepdim);

// Reduction Ops (Multiple dimensions)
mlc_tensor* mlc_sum_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_prod_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_mean_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_median_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_max_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_min_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_argmax_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);
mlc_tensor* mlc_argmin_(const mlc_tensor* a, const int64_t* dims, int64_t ndims, bool keepdim);

#endif
