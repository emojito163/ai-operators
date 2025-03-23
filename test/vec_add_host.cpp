// Copyright (c) 2021 The Houmo.ai Authors. All rights reserved.

//#ifdef _GLIBCXX_USE_FLOAT128
//#undef _GLIBCXX_USE_FLOAT128
//#endif

#include <cstdio>
#include <cstdlib>

#include <hdpl/hdpl_runtime.h>
#include <vec_kernels.h>

int main() {
  // NOTE data size must be multiple of 64 to fill cache line.
  int n = 64 * 4;
  auto a_host = (int8_t*)malloc(n * sizeof(int8_t));
  auto b_host = (int8_t*)malloc(n * sizeof(int8_t));
  auto c_host = (int8_t*)malloc(n * sizeof(int8_t));

  auto c_expect = (int8_t*)malloc(n * sizeof(int8_t));
  void* a_dev = nullptr;
  void* b_dev = nullptr;
  void* c_dev = nullptr;

  hdplMalloc(&a_dev, n * sizeof(int8_t));
  hdplMalloc(&b_dev, n * sizeof(int8_t));
  hdplMalloc(&c_dev, n * sizeof(int8_t));

  for (int i = 0; i < n; i++) {
    a_host[i] = 1;
    b_host[i] = 1;
    c_expect[i] = a_host[i] + b_host[i];
  }
  hdplMemcpy(a_dev, a_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);
  hdplMemcpy(b_dev, b_host, n * sizeof(int8_t), hdplMemcpyHostToDevice);

  // Launch kernel
  vec_add_kernel<<<1, 4>>>(a_dev, b_dev, c_dev, n);

  hdplStreamSynchronize(nullptr);

  hdplMemcpy(c_host, c_dev, n * sizeof(int8_t), hdplMemcpyDeviceToHost);

  bool res = true;
  for (int i = 0; i < n; i++) {
    if (c_host[i] != c_expect[i]) {
      printf("The result c[%d] is %d, the expect result should be %d.\n", i, c_host[i],
             c_expect[i]);
      res = false;
      break;
    }
  }

  free(a_host);
  free(b_host);
  free(c_host);
  free(c_expect);

  hdplFree(a_dev);
  hdplFree(b_dev);
  hdplFree(c_dev);

  if (res) {
    printf("=== vec add success ===\n");
    return 0;
  } else {
    printf("=== vec add failed ===\n");
    return -1;
  }
}
