#pragma once

#include <hdpl/intrinsic.h>

__global__ void vec_add_kernel(void*, void*, void*, int);
__global__ void vec_abs_kernel(void*, void*, int);
__global__ void vec_exp_kernel(void*, void*, int);

