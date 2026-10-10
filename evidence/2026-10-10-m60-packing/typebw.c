/* Matvec (n=1 and n=8) time and GB/s by weight type, same shape m=32768 k=2048, via ggml CUDA backend. */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }
int main(void) {
  ggml_backend_t be = ggml_backend_cuda_init(0);
  enum ggml_type types[] = { GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K };
  const char *tn[] = { "F16", "Q8_0", "Q4_K", "Q6_K" };
  int64_t k = 2048, m = 32768;
  for (int ti = 0; ti < 4; ti++) for (int nn = 1; nn <= 8; nn *= 8) {
    if (nn == 8 && ti > 2) continue;
    struct ggml_init_params ip = { .mem_size = 16u << 20, .mem_buffer = NULL, .no_alloc = true };
    struct ggml_context *ctx = ggml_init(ip);
    struct ggml_tensor *w = ggml_new_tensor_2d(ctx, types[ti], k, m);
    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, nn);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    size_t nb = ggml_nbytes(w);
    unsigned char *tmp = malloc(nb);
    for (size_t i = 0; i < nb; i++) tmp[i] = (unsigned char)(i * 2654435761u >> 13);
    ggml_backend_tensor_set(w, tmp, 0, nb); free(tmp);
    float *xf = calloc(k * nn, sizeof(float)); for (int64_t i = 0; i < k * nn; i++) xf[i] = 0.001f * (i % 97);
    ggml_backend_tensor_set(x, xf, 0, sizeof(float) * k * nn); free(xf);
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ggml_mul_mat(ctx, w, x));
    ggml_gallocr_alloc_graph(ga, gf);
    for (int i = 0; i < 10; i++) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    int it = 100; double t0 = now_us();
    for (int i = 0; i < it; i++) ggml_backend_graph_compute(be, gf);
    ggml_backend_synchronize(be);
    double us = (now_us() - t0) / it;
    printf("TYPE %-5s m=%d k=%lld n=%d  weights_MB=%8.2f  %9.1f us  %7.1f GB/s\n", tn[ti], (int)m, (long long)k, nn, nb / 1e6, us, nb / (us * 1e-6) / 1e9);
    ggml_gallocr_free(ga);
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
  }
  return 0;
}
