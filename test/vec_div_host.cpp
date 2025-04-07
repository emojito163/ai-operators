#include <cstddef>
#include <cstdlib>

#include <hdpl/hdpl_runtime.h>
#include <vec_kernels.h>

int main() noexcept {
  // NOTE data size must be multiple of 64 to fill cache line.
  constexpr ::std::size_t n{64 * 4};
  auto a_host = (int8_t*)malloc(n * sizeof(int8_t));
  auto b_host = (int8_t*)malloc(n * sizeof(int8_t));
  auto c_host = (int8_t*)malloc(n * sizeof(int8_t));

  auto c_expect = (int8_t*)malloc(n * sizeof(int8_t));
  int8_t* a_dev = nullptr;
  int8_t* b_dev = nullptr;
  int8_t* c_dev = nullptr;

  hdplMalloc(reinterpret_cast<void**>(&a_dev), n * sizeof(int8_t));
  hdplMalloc(reinterpret_cast<void**>(&b_dev), n * sizeof(int8_t));
  hdplMalloc(reinterpret_cast<void**>(&c_dev), n * sizeof(int8_t));

  for (::std::size_t i{}; i < n; ++i) {
    a_host[i] = i % 128;
    b_host[i] = (i + 128) % 128;
    if (b_host[i] == 0)
#if __has_cpp_attribute(unlikely)
        [[unlikely]]
#endif
        {
            b_host[i] = 1;
        }
    c_expect[i] = a_host[i] / b_host[i];
  }
  hdplMemcpy(a_dev, a_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);
  hdplMemcpy(b_dev, b_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);

  // Launch kernel
  vec_div_kernel<1, 4>(a_dev, b_dev, c_dev, n);

  hdplStreamSynchronize(nullptr);

  hdplMemcpy(c_host, c_dev, n * sizeof(int8_t), hdplMemcpyDeviceToHost);

  for (::std::size_t i{}; i < n; ++i) {
    if (c_host[i] != c_expect[i])
#if __has_cpp_attribute(unlikely)
        [[unlikely]]
#endif
    {
      __builtin_trap();
    }
  }

  free(a_host);
  free(b_host);
  free(c_host);
  free(c_expect);

  hdplFree(a_dev);
  hdplFree(b_dev);
  hdplFree(c_dev);

  return 0;
}

