/* Times LFM2-8B-A1B decode matmuls through the ggml CUDA backend, using the real
 * tensor types and shapes from the GGUF. usage: lfm2mm <n_tokens> <ids: same|distinct> */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NL 24
#define ITERS 200

static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec * 1e-3; }

static int is_attn(int l) { return l == 2 || l == 6 || l == 10 || l == 14 || l == 18 || l == 21; }

static struct ggml_context *ctx;
static ggml_backend_t be;
static ggml_gallocr_t galloc;
static struct ggml_tensor *W_g[NL], *W_u[NL], *W_d[NL];          /* dense FFN (layers 0,1) */
static struct ggml_tensor *W_in[NL], *W_out[NL];                 /* shortconv (all layers) */
static struct ggml_tensor *W_q[NL], *W_k[NL], *W_v[NL], *W_o[NL];/* attention (6 layers) */
static struct ggml_tensor *E_g, *E_u, *E_d;                      /* experts, shared by MoE layers */
static struct ggml_tensor *W_lm;
static struct ggml_tensor *x2048, *x7168, *xE, *xD, *ids;
static int nt;

static struct ggml_tensor *mk2(enum ggml_type t, int64_t a, int64_t b) { return ggml_new_tensor_2d(ctx, t, a, b); }
static struct ggml_tensor *mk3(enum ggml_type t, int64_t a, int64_t b, int64_t c) { return ggml_new_tensor_3d(ctx, t, a, b, c); }

/* weight-bytes read per call for a 2D/3D weight; for MoE only the active experts are read */
static double wbytes_mm(struct ggml_tensor *w) { return (double)ggml_nbytes(w); }
static double wbytes_moe(struct ggml_tensor *w, int used) { return (double)ggml_nbytes(w) * used / w->ne[2]; }

/* graph builders; each returns a graph that contains the requested category */
enum cat { C_DENSE_ATTN_CONV = 0, C_MOE_GATEUP = 1, C_MOE_DOWN = 2, C_LMHEAD = 3, C_ALL = 4 };

static struct ggml_cgraph *build(enum cat c) {
  struct ggml_cgraph *gf = ggml_new_graph(ctx);
  for (int l = 0; l < NL; l++) {
    int moe = l >= 2;
    if (c == C_DENSE_ATTN_CONV || c == C_ALL) {
      if (l < 2) {
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_g[l], x2048));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_u[l], x2048));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_d[l], x7168));
      }
      ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_in[l], x2048));
      ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_out[l], x2048));
      if (is_attn(l)) {
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_q[l], x2048));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_k[l], x2048));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_v[l], x2048));
        ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_o[l], x2048));
      }
    }
    if (moe && (c == C_MOE_GATEUP || c == C_ALL)) {
      ggml_build_forward_expand(gf, ggml_mul_mat_id(ctx, E_g, xE, ids));
      ggml_build_forward_expand(gf, ggml_mul_mat_id(ctx, E_u, xE, ids));
    }
    if (moe && (c == C_MOE_DOWN || c == C_ALL)) {
      ggml_build_forward_expand(gf, ggml_mul_mat_id(ctx, E_d, xD, ids));
    }
  }
  if (c == C_LMHEAD || c == C_ALL) ggml_build_forward_expand(gf, ggml_mul_mat(ctx, W_lm, x2048));
  return gf;
}

/* seconds-free timing: back-to-back graph computes, one sync at the end */
static double time_graph(struct ggml_cgraph *gf) {
  ggml_gallocr_alloc_graph(galloc, gf);
  for (int i = 0; i < 10; i++) ggml_backend_graph_compute(be, gf);
  ggml_backend_synchronize(be);
  double t0 = now_us();
  for (int i = 0; i < ITERS; i++) ggml_backend_graph_compute(be, gf);
  ggml_backend_synchronize(be);
  return (now_us() - t0) / ITERS;
}

static void fill(struct ggml_tensor *t) {
  size_t n = ggml_nbytes(t);
  unsigned char *buf = malloc(n);
  unsigned s = 12345u;
  for (size_t i = 0; i < n; i++) { s = s * 1103515245u + 12345u; buf[i] = (unsigned char)(s >> 16); }
  ggml_backend_tensor_set(t, buf, 0, n);
  free(buf);
}

int main(int argc, char **argv) {
  nt = argc > 1 ? atoi(argv[1]) : 1;
  int distinct = argc > 2 && strcmp(argv[2], "distinct") == 0;
  be = ggml_backend_cuda_init(0);
  if (!be) { fprintf(stderr, "cuda init failed\n"); return 1; }
  printf("tokens_per_step=%d expert_ids=%s\n", nt, distinct ? "distinct-per-token (union up to 32)" : "same-for-all-tokens (union 4)");

  struct ggml_init_params ip = { .mem_size = 256u << 20, .mem_buffer = NULL, .no_alloc = true };
  ctx = ggml_init(ip);
  for (int l = 0; l < NL; l++) {
    if (l < 2) { W_g[l] = mk2(GGML_TYPE_Q4_K, 2048, 7168); W_u[l] = mk2(GGML_TYPE_Q4_K, 2048, 7168); W_d[l] = mk2(GGML_TYPE_Q6_K, 7168, 2048); }
    W_in[l] = mk2(GGML_TYPE_Q4_K, 2048, 6144);
    W_out[l] = mk2(GGML_TYPE_Q4_K, 2048, 2048);
    if (is_attn(l)) {
      W_q[l] = mk2(GGML_TYPE_Q4_K, 2048, 2048); W_k[l] = mk2(GGML_TYPE_Q4_K, 2048, 512);
      W_v[l] = mk2(GGML_TYPE_Q4_K, 2048, 512);  W_o[l] = mk2(GGML_TYPE_Q4_K, 2048, 2048);
    }
  }
  E_g = mk3(GGML_TYPE_Q4_K, 2048, 1792, 32);
  E_u = mk3(GGML_TYPE_Q4_K, 2048, 1792, 32);
  E_d = mk3(GGML_TYPE_Q6_K, 1792, 2048, 32);
  W_lm = mk2(GGML_TYPE_Q6_K, 2048, 65536);
  x2048 = mk2(GGML_TYPE_F32, 2048, nt);
  x7168 = mk2(GGML_TYPE_F32, 7168, nt);
  xE = mk3(GGML_TYPE_F32, 2048, 1, nt);
  xD = mk3(GGML_TYPE_F32, 1792, 4, nt);
  ids = mk2(GGML_TYPE_I32, 4, nt);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
  if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
  printf("weights_buffer_MiB=%.0f\n", ggml_backend_buffer_get_size(buf) / 1048576.0);
  /* fill all weight tensors with pseudo-random bytes */
  for (int l = 0; l < NL; l++) {
    if (l < 2) { fill(W_g[l]); fill(W_u[l]); fill(W_d[l]); }
    fill(W_in[l]); fill(W_out[l]);
    if (is_attn(l)) { fill(W_q[l]); fill(W_k[l]); fill(W_v[l]); fill(W_o[l]); }
  }
  fill(E_g); fill(E_u); fill(E_d); fill(W_lm);
  fill(x2048); fill(x7168); fill(xE); fill(xD);
  int32_t *idv = malloc(sizeof(int32_t) * 4 * nt);
  for (int t = 0; t < nt; t++)
    for (int j = 0; j < 4; j++)
      idv[t * 4 + j] = distinct ? (((t * 4 + j) * 5 + 3) % 32) : (j == 0 ? 3 : j == 1 ? 17 : j == 2 ? 22 : 9);
  ggml_backend_tensor_set(ids, idv, 0, sizeof(int32_t) * 4 * nt);
  free(idv);

  galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));

  /* whole-token weight matmul sequence and per-category graphs */
  const char *names[] = { "dense+attn+conv matmuls (Q4_K/Q6_K)", "MoE gate+up mul_mat_id (Q4_K)", "MoE down mul_mat_id (Q6_K)", "lm_head (Q6_K)", "ALL matmuls in one graph" };
  for (int c = 0; c < 5; c++) {
    struct ggml_cgraph *gf = build((enum cat)c);
    double us = time_graph(gf);
    printf("CAT %-38s nodes=%3d  %9.1f us/step\n", names[c], ggml_graph_n_nodes(gf), us);
  }

  /* single-shape timings: one op per graph */
  struct { const char *name; struct ggml_tensor *w; struct ggml_tensor *x; int moe_used; } ops[] = {
    { "ffn_gate  Q4_K m=7168 k=2048 (dense L0)", W_g[0], x2048, 0 },
    { "ffn_down  Q6_K m=2048 k=7168 (dense L0)", W_d[0], x7168, 0 },
    { "shortconv in_proj  Q4_K m=6144 k=2048", W_in[2], x2048, 0 },
    { "shortconv out_proj Q4_K m=2048 k=2048", W_out[2], x2048, 0 },
    { "attn q    Q4_K m=2048 k=2048", W_q[2], x2048, 0 },
    { "attn k    Q4_K m=512  k=2048", W_k[2], x2048, 0 },
    { "attn o    Q4_K m=2048 k=2048", W_o[2], x2048, 0 },
    { "lm_head   Q6_K m=65536 k=2048", W_lm, x2048, 0 },
    { "MoE gate_exps Q4_K m=1792 k=2048 (4 of 32)", E_g, xE, 4 },
    { "MoE down_exps Q6_K m=2048 k=1792 (4 of 32)", E_d, xD, 4 },
  };
  for (int i = 0; i < 10; i++) {
    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    struct ggml_tensor *out = ops[i].moe_used
      ? (ops[i].w == E_d ? ggml_mul_mat_id(ctx, ops[i].w, xD, ids) : ggml_mul_mat_id(ctx, ops[i].w, xE, ids))
      : ggml_mul_mat(ctx, ops[i].w, ops[i].x);
    ggml_build_forward_expand(gf, out);
    double us = time_graph(gf);
    double bytes = ops[i].moe_used ? wbytes_moe(ops[i].w, ops[i].moe_used) : wbytes_mm(ops[i].w);
    printf("OP  %-44s weights_read_MB=%7.3f  %8.2f us  %7.1f GB/s\n", ops[i].name, bytes / 1e6, us, bytes / (us * 1e-6) / 1e9);
  }
  return 0;
}
