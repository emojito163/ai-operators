#include <cstdlib>
#include <cstdint>

#include <hdpl/hdpl_runtime.h>
#include <vec_kernels.hh>

int main() {
  // NOTE data size must be multiple of 64 to fill cache line.
    constexpr ::std::size_t n{64 * 4};
  auto a_host = reinterpret_cast<int8_t*>(malloc(n * sizeof(int8_t)));
  auto c_host = reinterpret_cast<int8_t*>(malloc(n * sizeof(int8_t)));

  int8_t* a_dev = nullptr;
  int8_t* c_dev = nullptr;

  hdplMalloc(reinterpret_cast<void**>(&a_dev), n * sizeof(int8_t));
  hdplMalloc(reinterpret_cast<void**>(&c_dev), n * sizeof(int8_t));

  for (::std::size_t i{}; i < n; ++i) {
    a_host[i] = i & 1 ? 1 : -1;
  }
  hdplMemcpy(a_dev, a_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);

  // Launch kernel
  vec_neg_kernel<1, 4>(a_dev, c_dev, n);

  hdplStreamSynchronize(nullptr);

  hdplMemcpy(c_host, c_dev, n * sizeof(int8_t), hdplMemcpyDeviceToHost);

  for (::std::size_t i{}; i < n; i++) {
    if (c_host[i] != (i & 1 ? -1 : 1))
#if __has_cpp_attribute(unlikely)
        [[unlikely]]
#endif
    {
      __builtin_trap();
    }
  }

  free(a_host);
  free(c_host);

  hdplFree(a_dev);
  hdplFree(c_dev);

  return 0;
}
