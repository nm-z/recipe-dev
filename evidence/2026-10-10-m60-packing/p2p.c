/* P2P copy latency/bandwidth between two visible devices (dev0 = die1, dev1 = die2). */
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#define CHK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s); return 1; } } while (0)
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
int main(void) {
  CHK(cuInit(0));
  CUdevice d[2]; CUcontext c[2]; CUdeviceptr p[2];
  int n; CHK(cuDeviceGetCount(&n)); printf("visible devices: %d\n", n);
  for (int i = 0; i < 2; i++) {
    CHK(cuDeviceGet(&d[i], i)); char nm[128]; CHK(cuDeviceGetName(nm, sizeof nm, d[i]));
    printf("dev%d: %s\n", i, nm);
    CHK(cuCtxCreate(&c[i], 0, d[i]));
    CHK(cuMemAlloc(&p[i], (size_t)512 << 20));
    CHK(cuCtxPopCurrent(NULL));
  }
  int can01 = 0, can10 = 0;
  CHK(cuDeviceCanAccessPeer(&can01, d[0], d[1])); CHK(cuDeviceCanAccessPeer(&can10, d[1], d[0]));
  printf("canAccessPeer 0->1=%d 1->0=%d\n", can01, can10);
  CHK(cuCtxPushCurrent(c[0])); CUresult r1 = cuCtxEnablePeerAccess(c[1], 0); printf("enablePeer 0->1: %d\n", (int)r1); CHK(cuCtxPopCurrent(NULL));
  CHK(cuCtxPushCurrent(c[1])); CUresult r2 = cuCtxEnablePeerAccess(c[0], 0); printf("enablePeer 1->0: %d\n", (int)r2); CHK(cuCtxPopCurrent(NULL));
  /* cuMemcpyPeer works regardless of peer-access enablement (staged if needed) */
  CUevent e0, e1; CHK(cuCtxPushCurrent(c[0])); CHK(cuEventCreate(&e0, 0)); CHK(cuEventCreate(&e1, 0)); CHK(cuCtxPopCurrent(NULL));
  size_t sz[] = { 4096, 16384, 1 << 20, 64 << 20, 256 << 20 };
  for (int s = 0; s < 5; s++) {
    size_t bytes = sz[s];
    int iters = bytes <= 16384 ? 2000 : (bytes <= (1 << 20) ? 500 : 20);
    CHK(cuCtxPushCurrent(c[0]));
    for (int w = 0; w < 5; w++) CHK(cuMemcpyPeer(p[1], c[1], p[0], c[0], bytes));
    CHK(cuEventRecord(e0, 0));
    double h0 = now_us();
    for (int i = 0; i < iters; i++) CHK(cuMemcpyPeer(p[1], c[1], p[0], c[0], bytes));
    double host = (now_us() - h0) / iters;
    CHK(cuEventRecord(e1, 0)); CHK(cuEventSynchronize(e1));
    CHK(cuCtxPopCurrent(NULL));
    float ms; CHK(cuEventElapsedTime(&ms, e0, e1));
    double per_us = ms * 1e3 / iters;
    printf("cuMemcpyPeer 0->1 %9zu B: %9.2f us/copy (GPU-event), %9.2f us (host, sync-free queue), %8.2f GB/s\n",
           bytes, per_us, host, bytes / (per_us * 1e-6) / 1e9);
  }
  /* latency: sync per copy, 4 KB */
  CHK(cuCtxPushCurrent(c[0]));
  double tot = 0; int L = 2000;
  for (int i = 0; i < L; i++) {
    double a = now_us();
    CHK(cuMemcpyPeer(p[1], c[1], p[0], c[0], 4096));
    CHK(cuCtxSynchronize());
    tot += now_us() - a;
  }
  CHK(cuCtxPopCurrent(NULL));
  printf("cuMemcpyPeer 4096 B with sync per copy (host round trip): %.2f us\n", tot / L);
  return 0;
}
