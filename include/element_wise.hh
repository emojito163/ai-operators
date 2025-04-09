#pragma once

#include <cstddef>
#include <cstdint>
#include <hdpl/intrinsic.h>

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_add_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                                ::std::int8_t* const __restrict c, ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        __builtin_add_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_add_kernel(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                           ::std::int8_t* const __restrict c, ::std::size_t const num) noexcept {
    ::details::vec_add_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_sub_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                                ::std::int8_t* const __restrict c, ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        __builtin_sub_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_sub_kernel(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                           ::std::int8_t* const __restrict c, ::std::size_t const num) noexcept {
    ::details::vec_sub_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_mul_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                                ::std::int8_t* const __restrict c, ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        __builtin_mul_overflow(a[offset + i], b[offset + i], c + offset + i);
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_mul_kernel(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                           ::std::int8_t* const __restrict c, ::std::size_t const num) noexcept {
    ::details::vec_mul_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_div_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                                ::std::int8_t* const __restrict c, ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        c[offset + i] = a[offset + i] / b[offset + i];
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_div_kernel(::std::int8_t const* const __restrict a, ::std::int8_t const* const __restrict b,
                           ::std::int8_t* const __restrict c, ::std::size_t const num) noexcept {
    ::details::vec_div_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, c, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_abs_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        *(b + offset + i) = __builtin_elementwise_abs(*(a + offset + i));
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_abs_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                           ::std::size_t const num) noexcept {
    ::details::vec_abs_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_neg_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        b[offset + i] = -a[offset + i];
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_neg_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                           ::std::size_t const num) noexcept {
    ::details::vec_neg_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_exp_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        if (a[offset + i] < 0) {
            b[offset + i] = 0;
        } else if (a[offset + i] == 0) {
            b[offset + i] = 1;
        } else if (a[offset + i] == 1) {
            b[offset + i] = 2;
        } else if (a[offset + i] == 2) {
            b[offset + i] = 7;
        } else if (a[offset + i] == 3) {
            b[offset + i] = 20;
        } else if (a[offset + i] == 4) {
            b[offset + i] = 54;
        } else {
            b[offset + i] = 127;
        }
    }

    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_exp_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                           ::std::size_t const num) noexcept {
    ::details::vec_exp_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum, bool ndebug>
__global__ void vec_sqrt_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                 ::std::size_t const num) {
    ::std::size_t num_per_proc{num / ProcNum};
    int proc_id = GetProcId();
    ::std::size_t const offset{proc_id * num_per_proc};

    for (::std::size_t i{}; i < num_per_proc; ++i) {
        if constexpr (!ndebug) {
            if (a[offset + i] < 0)
#if __has_cpp_attribute(unlikely)
                [[unlikely]]
#endif
            {
                __builtin_trap();
            }
        }
        /* Houmo device has problem with jump table
         * Therefore, do not use switch
         */
        if (a[offset + i] == 0) {
            b[offset + i] = 0;
        } else if (1 <= a[offset + i] && a[offset + i] < 4) {
            b[offset + i] = 1;
        } else if (4 <= a[offset + i] && a[offset + i] < 9) {
            b[offset + i] = 2;
        } else if (9 <= a[offset + i] && a[offset + i] < 16) {
            b[offset + i] = 3;
        } else if (16 <= a[offset + i] && a[offset + i] < 25) {
            b[offset + i] = 4;
        } else if (25 <= a[offset + i] && a[offset + i] < 36) {
            b[offset + i] = 5;
        } else if (36 <= a[offset + i] && a[offset + i] < 49) {
            b[offset + i] = 6;
        } else if (49 <= a[offset + i] && a[offset + i] < 64) {
            b[offset + i] = 7;
        } else if (64 <= a[offset + i] && a[offset + i] < 81) {
            b[offset + i] = 8;
        } else if (81 <= a[offset + i] && a[offset + i] < 100) {
            b[offset + i] = 9;
        } else if (100 <= a[offset + i] && a[offset + i] < 121) {
            b[offset + i] = 10;
        } else {
            b[offset + i] = 11;
        }
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

template<::std::size_t Kernel, ::std::size_t Tier, bool ndebug = false>
[[gnu::always_inline]]
inline void vec_sqrt_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                            ::std::size_t const num) noexcept {
    ::details::vec_sqrt_kernel_<Kernel * Tier, ndebug><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_min_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                ::std::size_t const num) {
    for (::std::size_t i{}; i < num; ++i) {
        if (a[i] < *b) {
            *b = a[i];
        }
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

/**
 * @brief Find minimum value in vector a and store it in b
 * @param[in]  a Vector a
 * @param[out] b Minimum value
 * @param[in]  num Number of elements in vector a
 */
template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_min_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                           ::std::size_t const num) noexcept {
    ::details::vec_min_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}

namespace details {

template<::std::size_t ProcNum>
__global__ void vec_max_kernel_(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                                ::std::size_t const num) {
    for (::std::size_t i{}; i < num; ++i) {
        if (a[i] > *b) {
            *b = a[i];
        }
    }
    RISCV_FENCE_I; // Flush cache
}

} // namespace details

/**
 * @brief Find max value in vector a and store it in b
 * @param[in]  a Vector a
 * @param[out] b max value
 * @param[in]  num Number of elements in vector a
 */
template<::std::size_t Kernel, ::std::size_t Tier>
[[gnu::always_inline]]
inline void vec_max_kernel(::std::int8_t const* const __restrict a, ::std::int8_t* const __restrict b,
                           ::std::size_t const num) noexcept {
    ::details::vec_max_kernel_<Kernel * Tier><<<Kernel, Tier>>>(a, b, num);
}
