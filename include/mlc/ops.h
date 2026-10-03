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

#endif
