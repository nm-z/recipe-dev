/* Single-die microbenchmarks via the CUDA driver API (driver 470 has no cudart 12). */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }

int main(void) {
  CHK(cuInit(0));
  CUdevice dev; CHK(cuDeviceGet(&dev, 0));
  char name[256]; CHK(cuDeviceGetName(name, sizeof name, dev));
  int smc, memclk, busw;
  CHK(cuDeviceGetAttribute(&smc, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
  CHK(cuDeviceGetAttribute(&memclk, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev));
  CHK(cuDeviceGetAttribute(&busw, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev));
  size_t total; CHK(cuDeviceTotalMem(&total, dev));
  printf("device: %s  SMs=%d  mem_clock_kHz=%d  bus_bits=%d  total_mem_MiB=%zu\n", name, smc, memclk, busw, total >> 20);
  printf("theoretical_peak_GB/s=%.1f\n", 2.0 * memclk * 1e3 * (busw / 8) / 1e9);

  CUcontext ctx; CHK(cuCtxCreate(&ctx, 0, dev));
  CUmodule mod; CHK(cuModuleLoad(&mod, "bw.cubin"));
  CUfunction fEmpty, fRead, fCopy;
  CHK(cuModuleGetFunction(&fEmpty, mod, "k_empty"));
  CHK(cuModuleGetFunction(&fRead, mod, "k_read"));
  CHK(cuModuleGetFunction(&fCopy, mod, "k_copy"));

  CUevent e0, e1; CHK(cuEventCreate(&e0, 0)); CHK(cuEventCreate(&e1, 0));

  /* ---- 1. device memory bandwidth ---- */
  size_t sizes[] = { (size_t)1 << 30, (size_t)256 << 20, (size_t)32 << 20 };
  CUdeviceptr buf_a, buf_b, outp;
  CHK(cuMemAlloc(&buf_a, sizes[0])); CHK(cuMemAlloc(&buf_b, sizes[0])); CHK(cuMemAlloc(&outp, 4));
  CHK(cuMemsetD8(buf_a, 1, sizes[0])); CHK(cuMemsetD8(buf_b, 2, sizes[0]));
  for (int si = 0; si < 3; si++) {
    size_t bytes = sizes[si], n16 = bytes / 16;
    unsigned blocks = 2048, threads = 256;
    void *argsR[] = { &buf_a, &outp, &n16 };
    void *argsC[] = { &buf_b, &buf_a, &n16 };
    int iters = 20;
    for (int w = 0; w < 3; w++) CHK(cuLaunchKernel(fRead, blocks, 1, 1, threads, 1, 1, 0, 0, argsR, 0));
    CHK(cuCtxSynchronize());
    CHK(cuEventRecord(e0, 0));
    for (int i = 0; i < iters; i++) CHK(cuLaunchKernel(fRead, blocks, 1, 1, threads, 1, 1, 0, 0, argsR, 0));
    CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
    float ms; CHK(cuEventElapsedTime(&ms, e0, e1));
    printf("READ   kernel  %5zu MiB: %.1f GB/s\n", bytes >> 20, (double)bytes * iters / (ms * 1e-3) / 1e9);

    for (int w = 0; w < 3; w++) CHK(cuLaunchKernel(fCopy, blocks, 1, 1, threads, 1, 1, 0, 0, argsC, 0));
    CHK(cuCtxSynchronize());
    CHK(cuEventRecord(e0, 0));
    for (int i = 0; i < iters; i++) CHK(cuLaunchKernel(fCopy, blocks, 1, 1, threads, 1, 1, 0, 0, argsC, 0));
    CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
    CHK(cuEventElapsedTime(&ms, e0, e1));
    printf("COPY   kernel  %5zu MiB: %.1f GB/s (read+write)\n", bytes >> 20, 2.0 * bytes * iters / (ms * 1e-3) / 1e9);

    CHK(cuEventRecord(e0, 0));
    for (int i = 0; i < iters; i++) CHK(cuMemcpyDtoD(buf_b, buf_a, bytes));
    CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
    CHK(cuEventElapsedTime(&ms, e0, e1));
    printf("COPY   cuMemcpyDtoD %5zu MiB: %.1f GB/s (read+write)\n", bytes >> 20, 2.0 * bytes * iters / (ms * 1e-3) / 1e9);
  }

  /* ---- 4. empty kernel launch / sync latency and throughput ---- */
  void *noargs[] = { 0 };
  for (int w = 0; w < 100; w++) { CHK(cuLaunchKernel(fEmpty, 1, 1, 1, 32, 1, 1, 0, 0, noargs, 0)); }
  CHK(cuCtxSynchronize());
  int L = 5000;
  double t0 = now_us();
  for (int i = 0; i < L; i++) { CHK(cuLaunchKernel(fEmpty, 1, 1, 1, 32, 1, 1, 0, 0, noargs, 0)); CHK(cuCtxSynchronize()); }
  double lat = (now_us() - t0) / L;
  printf("EMPTY launch+sync round trip (host timed): %.2f us per kernel\n", lat);

  int N = 100000;
  CHK(cuEventRecord(e0, 0));
  double h0 = now_us();
  for (int i = 0; i < N; i++) CHK(cuLaunchKernel(fEmpty, 1, 1, 1, 32, 1, 1, 0, 0, noargs, 0));
  double hostlaunch = (now_us() - h0) / N;
  CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
  float ms; CHK(cuEventElapsedTime(&ms, e0, e1));
  printf("EMPTY launch throughput, no sync: host enqueue %.2f us/launch; GPU-side back-to-back %.2f us/kernel (events)\n",
         hostlaunch, ms * 1e3 / N);

  /* empty-kernel launch with 256 threads x 1024 blocks (typical size) */
  CHK(cuEventRecord(e0, 0));
  for (int i = 0; i < N / 10; i++) CHK(cuLaunchKernel(fEmpty, 1024, 1, 1, 256, 1, 1, 0, 0, noargs, 0));
  CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
  CHK(cuEventElapsedTime(&ms, e0, e1));
  printf("EMPTY 1024x256 grid back-to-back: %.2f us/kernel (events)\n", ms * 1e3 / (N / 10));

  CHK(cuMemFree(buf_a)); CHK(cuMemFree(buf_b)); CHK(cuMemFree(outp));
  CHK(cuCtxDestroy(ctx));
  return 0;
}
