#pragma once

#include <cstddef>
#include <cstdint>
#include <hdpl/intrinsic.h>

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_add_kernel_(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i = 0; i < num_per_proc; ++i) {
        __builtin_add_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_add_kernel(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) noexcept {
    ::details::vec_add_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_sub_kernel_(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i = 0; i < num_per_proc; ++i) {
        __builtin_sub_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_sub_kernel(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) noexcept {
    ::details::vec_sub_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_mul_kernel_(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i = 0; i < num_per_proc; ++i) {
        __builtin_mul_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_mul_kernel(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) noexcept {
    ::details::vec_mul_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_div_kernel_(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i = 0; i < num_per_proc; ++i) {
        c[offset + i] = a[offset + i] / b[offset + i];
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_div_kernel(int8_t* a, int8_t* b, int8_t* c, ::std::size_t num) noexcept {
    ::details::vec_div_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_abs_kernel_(int8_t* a, int8_t* b, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i = 0; i < num_per_proc; ++i) {
        *(b + offset + i) = __builtin_elementwise_abs(*(a + offset + i));
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_abs_kernel(int8_t* a, int8_t* b, ::std::size_t num) noexcept {
    ::details::vec_abs_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_exp_kernel_(int8_t* a, int8_t* b, ::std::size_t num) {
    int num_per_proc = num / ProcNum;
    int proc_id = GetProcId();
    int offset = proc_id * num_per_proc;

    for (int i{}; i < num_per_proc; ++i) {
        if (a[offset + i] < 0) {
            b[offset + i] = 0;
        }
        switch (a[offset + i]) {
        case 0:
            b[offset + i] = 1;
        case 1:
            b[offset + i] = 2;
        case 2:
            b[offset + i] = 7;
        case 3:
            b[offset + i] = 20;
        case 4:
            b[offset + i] = 54;
        default:
            b[offset + i] = 127;
        }
    }
    RISCV_FENCE_I;  // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
inline void vec_exp_kernel(int8_t* a, int8_t* b, ::std::size_t num) noexcept {
    ::details::vec_exp_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

