/* Q4_K matvec benchmark: stock ggml CUDA kernel vs kernels A (XMAD) and B (binning), same process, same data.
 * usage: kb [m k]...   (device 0 as seen via CUDA_VISIBLE_DEVICES)  env: NITER, ONLY=A|B|S, NWLIST="2 4 8 16" */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#define CHK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r_, &s); \
  fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #x, s); exit(1); } } while (0)

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
static uint64_t rs = 88172645463325252ull;
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)(rs >> 16); }
static double frnd(void) { return (rnd() & 0xffffff) / 16777216.0; }

typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } blk_t;
static uint16_t f2h(float f) { _Float16 h = (_Float16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static float h2f(uint16_t u) { _Float16 h; memcpy(&h, &u, 2); return (float)h; }

static void get_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
  if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
  else { *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

/* host reference: out[r] = sum_i dequant(W[r][i]) * xv[i]  (double) */
static void ref_matvec(const blk_t *W, int m, int k, const double *xv, double *out) {
  int nb = k / 256;
  for (int r = 0; r < m; r++) {
    double acc = 0;
    for (int b = 0; b < nb; b++) {
      const blk_t *B = &W[(size_t)r * nb + b];
      double d = h2f(B->d), dm = h2f(B->dmin);
      for (int j = 0; j < 8; j++) {
        uint8_t sc, mn; get_scale_min(j, B->scales, &sc, &mn);
        int g = j >> 1, hi = j & 1;
        double s = 0;
        for (int l = 0; l < 32; l++) {
          int q = hi ? (B->qs[32 * g + l] >> 4) : (B->qs[32 * g + l] & 15);
          s += (d * sc * q - dm * mn) * xv[b * 256 + j * 32 + l];
        }
        acc += s;
      }
    }
    out[r] = acc;
  }
}
/* host q8_1 quantization identical to ggml's (float d, roundf) -> dequantized x */
static void q8_deq(const float *x, int k, double *xd) {
  for (int b = 0; b < k / 32; b++) {
    float amax = 0; for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(x[b * 32 + i]));
    float d = amax / 127.0f; float dh = h2f(f2h(d));
    for (int i = 0; i < 32; i++) { int q = amax == 0 ? 0 : (int)roundf(x[b * 32 + i] / d); xd[b * 32 + i] = (double)dh * q; }
  }
}

static void cmp(const char *name, const float *a, const float *ref, int m, double *outmaxabs) {
  double mx = 0; for (int i = 0; i < m; i++) mx = fmax(mx, fabs(ref[i]));
  double e_abs = 0, e_rel = 0; int nrel = 0;
  for (int i = 0; i < m; i++) {
    double e = fabs((double)a[i] - ref[i]); if (e > e_abs) e_abs = e;
    if (fabs(ref[i]) > 0.05 * mx) { double r = e / fabs(ref[i]); if (r > e_rel) e_rel = r; nrel++; }
  }
  printf("  CORR %-22s max|err|/max|ref| = %.3e   max elementwise rel err (|ref|>5%% of max, n=%d) = %.3e\n", name, e_abs / mx, nrel, e_rel);
  if (outmaxabs) *outmaxabs = e_abs / mx;
}

int main(int argc, char **argv) {
  int shapes[16][2]; int ns = 0;
  if (argc >= 3) { for (int i = 1; i + 1 < argc && ns < 16; i += 2) { shapes[ns][0] = atoi(argv[i]); shapes[ns][1] = atoi(argv[i + 1]); ns++; } }
  else { int d[4][2] = { {4096, 14336}, {7168, 2048}, {6144, 2048}, {2048, 2048} }; memcpy(shapes, d, sizeof d); ns = 4; }
  int niter = getenv("NITER") ? atoi(getenv("NITER")) : 300;
  const char *only = getenv("ONLY");
  int nwl[4] = { 2, 4, 8, 16 }, nnw = 4;
  if (getenv("NWLIST")) { nnw = sscanf(getenv("NWLIST"), "%d %d %d %d", &nwl[0], &nwl[1], &nwl[2], &nwl[3]); }

  ggml_backend_t be = ggml_backend_cuda_init(0);
  if (!be) { fprintf(stderr, "cuda init failed\n"); return 1; }
  CHK(cuInit(0)); CUdevice dev0; CHK(cuDeviceGet(&dev0, 0)); CUcontext ctx; CHK(cuDevicePrimaryCtxRetain(&ctx, dev0)); CHK(cuCtxSetCurrent(ctx));
  CUdevice dev; CHK(cuCtxGetDevice(&dev)); char dn[128]; CHK(cuDeviceGetName(dn, 128, dev)); printf("device: %s\n", dn);
  CUmodule mod; CHK(cuModuleLoad(&mod, "kern.cubin"));
  CUfunction fprep; CHK(cuModuleGetFunction(&fprep, mod, "prep"));
  CUfunction fK[4][17]; memset(fK, 0, sizeof fK); CUfunction fprepf; CHK(cuModuleGetFunction(&fprepf, mod, "prepf")); const char *kn = "ABCF", *kl[4] = {"A xmad","B bins","C fp32-magic","C2 fp32+fadd"};
  for (int i = 0; i < nnw; i++) { char n[16]; int w = nwl[i]; for (int kd = 0; kd < 4; kd++) { snprintf(n, 16, "k%c_%d", kn[kd], w); if (cuModuleGetFunction(&fK[kd][w], mod, n) != CUDA_SUCCESS) fK[kd][w] = 0; } }

  for (int si = 0; si < ns; si++) {
    int m = shapes[si][0], k = shapes[si][1];
    printf("=== shape m=%d k=%d (%dx%d) ===\n", m, k, m, k);
    struct ggml_init_params ip = { .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *gc = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(gc, GGML_TYPE_Q4_K, k, m);
    struct ggml_tensor *x = ggml_new_tensor_2d(gc, GGML_TYPE_F32, k, 1);
    int R = 8; struct ggml_tensor *outs[8];
    struct ggml_cgraph *g1 = ggml_new_graph(gc), *g8 = ggml_new_graph_custom(gc, 64, false);
    struct ggml_tensor *o1 = ggml_mul_mat(gc, w, x); ggml_build_forward_expand(g1, o1);
    for (int i = 0; i < R; i++) { outs[i] = ggml_mul_mat(gc, w, x); ggml_build_forward_expand(g8, outs[i]); }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(gc, be);
    size_t nb = ggml_nbytes(w); int nbk = m * (k / 256);
    blk_t *hw = malloc(nb);
    for (int i = 0; i < nbk; i++) {
      hw[i].d = f2h(0.002f + 0.03f * (float)frnd()); hw[i].dmin = f2h(0.001f + 0.02f * (float)frnd());
      for (int j = 0; j < 12; j++) hw[i].scales[j] = rnd() & 255;
      for (int j = 0; j < 128; j++) hw[i].qs[j] = rnd() & 255;
    }
    float *hx = malloc(4 * k);
    for (int i = 0; i < k; i++) { double u = frnd() * 2 - 1; hx[i] = (float)(u * (rnd() % 64 == 0 ? 8.0 : 1.0)); }
    ggml_backend_tensor_set(w, hw, 0, nb); ggml_backend_tensor_set(x, hx, 0, 4 * k);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, g1);
    ggml_backend_graph_compute(be, g1); ggml_backend_synchronize(be);
    float *ystock = malloc(4 * m); ggml_backend_tensor_get(o1, ystock, 0, 4 * m);
    double wGB = (double)nb;

    /* references */
    double *xt = malloc(8 * k), *xq = malloc(8 * k), *rt = malloc(8 * m), *rq = malloc(8 * m);
    for (int i = 0; i < k; i++) xt[i] = hx[i];
    q8_deq(hx, k, xq);
    ref_matvec(hw, m, k, xt, rt); ref_matvec(hw, m, k, xq, rq);
    float *frt = malloc(4 * m), *frq = malloc(4 * m);
    for (int i = 0; i < m; i++) { frt[i] = (float)rt[i]; frq[i] = (float)rq[i]; }
    printf("  weights %.3f MB\n", wGB / 1e6);
    cmp("stock vs f64(x f32)", ystock, frt, m, NULL);
    cmp("stock vs f64(x q8_1)", ystock, frq, m, NULL);

    /* ---- stock timing ---- */
    if (!only || only[0] == 'S') {
      for (int i = 0; i < 20; i++) ggml_backend_graph_compute(be, g1);
      ggml_backend_synchronize(be); double t0 = now_us();
      for (int i = 0; i < niter; i++) ggml_backend_graph_compute(be, g1);
      ggml_backend_synchronize(be); double us1 = (now_us() - t0) / niter;
      ggml_gallocr_free(ga); ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
      ggml_gallocr_alloc_graph(ga, g8);
      for (int i = 0; i < 5; i++) ggml_backend_graph_compute(be, g8);
      ggml_backend_synchronize(be); t0 = now_us();
      for (int i = 0; i < niter / 4; i++) ggml_backend_graph_compute(be, g8);
      ggml_backend_synchronize(be); double us8 = (now_us() - t0) / (niter / 4) / R;
      printf("  TIME stock(ggml mul_mat, 1-node graph)  %8.2f us  %7.1f GB/s\n", us1, wGB / (us1 * 1e-6) / 1e9);
      printf("  TIME stock(ggml mul_mat, 8-node graph/8)%8.2f us  %7.1f GB/s\n", us8, wGB / (us8 * 1e-6) / 1e9);
    }

    /* ---- mine ---- */
    CUdeviceptr dW = (CUdeviceptr)w->data, dx, dy16, dy32, dds, dout;
    CHK(cuMemAlloc(&dx, 4 * k)); CHK(cuMemAlloc(&dy16, 2 * k)); CHK(cuMemAlloc(&dy32, 4 * k));
    CHK(cuMemAlloc(&dds, 8 * (k / 32))); CHK(cuMemAlloc(&dout, 4 * m));
    CHK(cuMemcpyHtoD(dx, hx, 4 * k));
    int nblk = k / 32; unsigned pgrid = (nblk + 7) / 8;
    void *pargs[] = { &dx, &nblk, &dy16, &dy32, &dds };
    void *pargsf[] = { &dx, &nblk, &dds };
    CUevent e0, e1; CHK(cuEventCreate(&e0, 0)); CHK(cuEventCreate(&e1, 0));
    for (int kind = 0; kind < 4; kind++) {
      if (only && only[0] != kn[kind]) continue;
      CUfunction pf = kind < 2 ? fprep : fprepf; void **pa = kind < 2 ? pargs : pargsf; unsigned pg = pgrid; 
      for (int ii = 0; ii < nnw; ii++) {
        int nw = nwl[ii]; CUfunction f = fK[kind][nw]; if (!f) continue;
        void *kargs[] = { &dW, kind == 0 ? (void *)&dy16 : kind == 1 ? (void *)&dy32 : (void *)&dx, &dds, &dout, &k };
        unsigned grid = m / nw;
        CHK(cuMemsetD32(dout, 0, m));
        CHK(cuLaunchKernel(pf, pg, 1, 1, 256, 1, 1, 0, 0, pa, 0));
        CHK(cuLaunchKernel(f, grid, 1, 1, 32 * nw, 1, 1, 0, 0, kargs, 0)); CHK(cuCtxSynchronize());
        float *yo = malloc(4 * m); CHK(cuMemcpyDtoH(yo, dout, 4 * m));
        char nm[64]; snprintf(nm, 64, "%s nw=%d vs stock", kl[kind], nw);
        printf("  [%s nw=%d]\n", kl[kind], nw);
        cmp(nm, yo, ystock, m, NULL);
        cmp("  vs f64(x q8_1)", yo, frq, m, NULL);
        cmp("  vs f64(x f32)", yo, frt, m, NULL);
        free(yo);
        for (int i = 0; i < 20; i++) { cuLaunchKernel(pf, pg, 1, 1, 256, 1, 1, 0, 0, pa, 0); cuLaunchKernel(f, grid, 1, 1, 32 * nw, 1, 1, 0, 0, kargs, 0); }
        CHK(cuCtxSynchronize());
        double t0 = now_us();
        for (int i = 0; i < niter; i++) { cuLaunchKernel(pf, pg, 1, 1, 256, 1, 1, 0, 0, pa, 0); cuLaunchKernel(f, grid, 1, 1, 32 * nw, 1, 1, 0, 0, kargs, 0); }
        CHK(cuCtxSynchronize()); double us_pair = (now_us() - t0) / niter;
        t0 = now_us();
        for (int i = 0; i < niter; i++) cuLaunchKernel(f, grid, 1, 1, 32 * nw, 1, 1, 0, 0, kargs, 0);
        CHK(cuCtxSynchronize()); double us_k = (now_us() - t0) / niter;
        t0 = now_us();
        for (int i = 0; i < niter; i++) cuLaunchKernel(pf, pg, 1, 1, 256, 1, 1, 0, 0, pa, 0);
        CHK(cuCtxSynchronize()); double us_p = (now_us() - t0) / niter;
        printf("  TIME %-12s nw=%-2d prep+kernel %8.2f us %7.1f GB/s | kernel only %8.2f us %7.1f GB/s | prep only %.2f us\n",
               kl[kind], nw, us_pair, wGB / (us_pair * 1e-6) / 1e9, us_k, wGB / (us_k * 1e-6) / 1e9, us_p);
      }
    }
    cuMemFree(dx); cuMemFree(dy16); cuMemFree(dy32); cuMemFree(dds); cuMemFree(dout);
    ggml_gallocr_free(ga); ggml_backend_buffer_free(buf); ggml_free(gc);
    free(hw); free(hx); free(ystock); free(xt); free(xq); free(rt); free(rq); free(frt); free(frq);
  }
  return 0;
}
