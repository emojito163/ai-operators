#include <cstdlib>
#include <cstdint>

#include <hdpl/hdpl_runtime.h>
#include <vec_kernels.hh>

int main() {
  // NOTE data size must be multiple of 64 to fill cache line.
  constexpr ::std::size_t n{64 * 4};
  int8_t a_host[n]{};
  int8_t c_host[n]{};

  int8_t* a_dev = nullptr;
  int8_t* c_dev = nullptr;

  hdplMalloc(reinterpret_cast<void**>(&a_dev), n * sizeof(int8_t));
  hdplMalloc(reinterpret_cast<void**>(&c_dev), n * sizeof(int8_t));

  for (::std::size_t i{}; i < n; ++i) {
    a_host[i] = i % 128;
  }
  hdplMemcpy(a_dev, a_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);

  // Launch kernel
  vec_min_kernel<1, 4>(a_dev, c_dev, n);

  hdplStreamSynchronize(nullptr);

  hdplMemcpy(c_host, c_dev, n * sizeof(int8_t), hdplMemcpyDeviceToHost);

  if (c_host[0] != 0)
#if __has_cpp_attribute(unlikely)
    [[unlikely]]
#endif
  {
    __builtin_trap();
  }

  hdplFree(a_dev);
  hdplFree(c_dev);

  return 0;
}
