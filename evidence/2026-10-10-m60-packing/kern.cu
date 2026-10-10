// Q4_K decode matvec kernels for Maxwell sm_52 (no dp4a).  A = XMAD 16x16 path, B = binning path.
#include <cuda_fp16.h>
// Unit = (super-block, group g): 32 bytes of qs = sub-blocks 2g (low nibbles) and 2g+1 (high nibbles).
// One warp = one output row; lane l of tile t handles super-block 8t+(l>>2), group (l&3).
#include <stdint.h>
typedef unsigned int u32;
#define TU 32
#define M2 0x000F000Fu

// ---------------- activation prep: f32 -> q8_1-equivalent -----------------
// y16: per 64-value unit, 32 words. sub-block A then B; within a sub-block, word 2p = (y[4p+0], y[4p+2]) as int16 halves,
// word 2p+1 = (y[4p+1], y[4p+3]).  y32: plain sign-extended ints.  dsf[sb] = {d8 (fp16-rounded like stock), d8*sum(q)}.
extern "C" __global__ void prep(const float* __restrict__ x, int nblk, u32* __restrict__ y16, int* __restrict__ y32, float2* __restrict__ dsf)
{
  const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (b >= nblk) return;
  const float v = x[b * 32 + lane];
  float amax = fabsf(v);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  const float d = amax / 127.0f;
  const int q = amax == 0.0f ? 0 : (int)roundf(v / d);
  int s = q;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  y32[b * 32 + lane] = q;
  // y16 permuted: lane = 4p+c ; c in 0..3. word index 2p + (c&1), half (c>>1)
  const int p = lane >> 2, c = lane & 3;
  const int widx = b * 16 + 2 * p + (c & 1);
  // gather the partner (c^2) value into the even-c... : lanes c=0,1 write words (lo=own, hi=partner c+2)
  const int partner = __shfl_xor_sync(0xffffffffu, q, 2);
  if (c < 2) y16[widx] = ((u32)(q & 0xffff)) | ((u32)(partner & 0xffff) << 16);
  if (lane == 0) { const float d8 = __half2float(__float2half(d)); dsf[b] = make_float2(d8, d8 * (float)s); }
}

static __device__ __forceinline__ int mad2(u32 m, u32 y, int acc) {
  asm("{ .reg .b16 ml,mh,yl,yh; mov.b32 {ml,mh}, %1; mov.b32 {yl,yh}, %2; mad.wide.s16 %0, ml, yl, %0; mad.wide.s16 %0, mh, yh, %0; }"
      : "+r"(acc) : "r"(m), "r"(y));
  return acc;
}

static __device__ __forceinline__ void decode_scales(u32 S0, u32 S1, u32 S2, int g, int& scA, int& scB, int& mA, int& mB) {
  const int a = (g & 1) * 16;
  const u32 x0 = (S0 >> a) & 0xffffu, x1 = (S1 >> a) & 0xffffu, x2 = (S2 >> a) & 0xffffu;
  u32 sp, mp;
  if (g < 2) { sp = x0 & 0x3f3fu; mp = x1 & 0x3f3fu; }
  else { sp = (x2 & 0x0f0fu) | ((x0 & 0xc0c0u) >> 2); mp = ((x2 >> 4) & 0x0f0fu) | ((x1 & 0xc0c0u) >> 2); }
  scA = sp & 0xff; scB = sp >> 8; mA = mp & 0xff; mB = mp >> 8;
}

static __device__ __forceinline__ float finish_unit(u32 dd, int dotA, int dotB, int scA, int scB, int mA, int mB, float2 dA, float2 dB) {
  const float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xffff)));
  const float dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
  const float pd = fmaf((float)scA * dA.x, (float)dotA, (float)scB * dB.x * (float)dotB);
  const float pm = fmaf((float)mA, dA.y, (float)mB * dB.y);
  return fmaf(d, pd, -dmin * pm);
}

static __device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

// ======================= kernel A: XMAD =======================
#define YS16 36   // words per unit in smem (32 + 4 pad -> conflict-free LDS.128)
template <int NW>
static __device__ __forceinline__ void kernA(const uint8_t* __restrict__ W, const u32* __restrict__ y16, const float2* __restrict__ dsf, float* __restrict__ out, int k)
{
  constexpr int NT = NW * 32;
  constexpr int NY = (TU * 8 + NT - 1) / NT;           // uint4 per thread per tile
  __shared__ __align__(16) u32 ys[2][TU * YS16];
  __shared__ float2 dss[2][64];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.x * NW + warp;
  const int nsb = k >> 8, nt = nsb >> 3;
  const uint8_t* rowp = W + (size_t)row * nsb * 144;
  const int sbl = lane >> 2, g = lane & 3;
  const uint4* y4 = (const uint4*)y16;

  uint4 yr[NY]; float2 dr = make_float2(0, 0);
  // stage tile 0
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 8) yr[i] = y4[ii]; }
  if (tid < 64) dr = dsf[tid];
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 8) *(uint4*)&ys[0][(ii >> 3) * YS16 + (ii & 7) * 4] = yr[i]; }
  if (tid < 64) dss[0][tid] = dr;

  const uint8_t* bp = rowp + sbl * 144;
  uint4 q0 = *(const uint4*)(bp + 16 + 32 * g), q1 = *(const uint4*)(bp + 32 + 32 * g);
  u32 S0 = *(const u32*)(bp + 4), S1 = *(const u32*)(bp + 8), S2 = *(const u32*)(bp + 12), dd = *(const u32*)bp;
  __syncthreads();
  float facc = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int buf = t & 1;
    uint4 n0, n1; u32 nS0, nS1, nS2, ndd;
    if (t + 1 < nt) {
      const uint8_t* np = rowp + ((t + 1) * 8 + sbl) * 144;
      n0 = *(const uint4*)(np + 16 + 32 * g); n1 = *(const uint4*)(np + 32 + 32 * g);
      nS0 = *(const u32*)(np + 4); nS1 = *(const u32*)(np + 8); nS2 = *(const u32*)(np + 12); ndd = *(const u32*)np;
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 8) yr[i] = y4[(size_t)(t + 1) * TU * 8 + ii]; }
      if (tid < 64) dr = dsf[(t + 1) * 64 + tid];
    }
    const u32* yu = &ys[buf][lane * YS16];
    u32 w[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
    int aA0 = 0, aA1 = 0, aB0 = 0, aB1 = 0;
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
      const uint4 ya = *(const uint4*)(yu + 4 * h);        // sub-block A words 4h..4h+3
      const uint4 yb = *(const uint4*)(yu + 16 + 4 * h);   // sub-block B
      #pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const u32 ww = w[2 * h + jj];
        const u32 yA0 = jj ? ya.z : ya.x, yA1 = jj ? ya.w : ya.y, yB0 = jj ? yb.z : yb.x, yB1 = jj ? yb.w : yb.y;
        aA0 = mad2(ww & M2, yA0, aA0);
        aB0 = mad2((ww >> 4) & M2, yB0, aB0);
        aA1 = mad2((ww >> 8) & M2, yA1, aA1);
        aB1 = mad2((ww >> 12) & M2, yB1, aB1);
      }
    }
    int scA, scB, mA, mB; decode_scales(S0, S1, S2, g, scA, scB, mA, mB);
    const float2 dA = dss[buf][sbl * 8 + 2 * g], dB = dss[buf][sbl * 8 + 2 * g + 1];
    facc += finish_unit(dd, aA0 + aA1, aB0 + aB1, scA, scB, mA, mB, dA, dB);
    if (t + 1 < nt) {
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 8) *(uint4*)&ys[buf ^ 1][(ii >> 3) * YS16 + (ii & 7) * 4] = yr[i]; }
      if (tid < 64) dss[buf ^ 1][tid] = dr;
      q0 = n0; q1 = n1; S0 = nS0; S1 = nS1; S2 = nS2; dd = ndd;
    }
    __syncthreads();
  }
  facc = warp_sum(facc);
  if (lane == 0) out[row] = facc;
}

// ======================= kernel B: binning =======================
#define YS32 68
#ifndef NSET
#define NSET 2
#endif
template <int NW>
static __device__ __forceinline__ void kernB(const uint8_t* __restrict__ W, const int* __restrict__ y32, const float2* __restrict__ dsf, float* __restrict__ out, int k)
{
  constexpr int NT = NW * 32;
  constexpr int NY = (TU * 16 + NT - 1) / NT;
  __shared__ __align__(16) int ys[2][TU * YS32];
  __shared__ float2 dss[2][64];
  __shared__ short bins[NSET][NW][16 * 32];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int row = blockIdx.x * NW + warp;
  const int nsb = k >> 8, nt = nsb >> 3;
  const uint8_t* rowp = W + (size_t)row * nsb * 144;
  const int sbl = lane >> 2, g = lane & 3;
  const uint4* y4 = (const uint4*)y32;
  #pragma unroll
  for (int s = 0; s < NSET; ++s) for (int v = 0; v < 16; ++v) bins[s][warp][v * 32 + lane] = 0;

  uint4 yr[NY]; float2 dr = make_float2(0, 0);
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) yr[i] = y4[ii]; }
  if (tid < 64) dr = dsf[tid];
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) *(uint4*)&ys[0][(ii >> 4) * YS32 + (ii & 15) * 4] = yr[i]; }
  if (tid < 64) dss[0][tid] = dr;

  const uint8_t* bp = rowp + sbl * 144;
  uint4 q0 = *(const uint4*)(bp + 16 + 32 * g), q1 = *(const uint4*)(bp + 32 + 32 * g);
  u32 S0 = *(const u32*)(bp + 4), S1 = *(const u32*)(bp + 8), S2 = *(const u32*)(bp + 12), dd = *(const u32*)bp;
  __syncthreads();
  float facc = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int buf = t & 1;
    uint4 n0, n1; u32 nS0, nS1, nS2, ndd;
    if (t + 1 < nt) {
      const uint8_t* np = rowp + ((t + 1) * 8 + sbl) * 144;
      n0 = *(const uint4*)(np + 16 + 32 * g); n1 = *(const uint4*)(np + 32 + 32 * g);
      nS0 = *(const u32*)(np + 4); nS1 = *(const u32*)(np + 8); nS2 = *(const u32*)(np + 12); ndd = *(const u32*)np;
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) yr[i] = y4[(size_t)(t + 1) * TU * 16 + ii]; }
      if (tid < 64) dr = dsf[(t + 1) * 64 + tid];
    }
    const int* yu = &ys[buf][lane * YS32];
    u32 w[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
    int dot[2];
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int4 y = *(const int4*)(yu + pass * 32 + 4 * j);
        const int yv[4] = { y.x, y.y, y.z, y.w };
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          const u32 q = (w[j] >> (8 * c + 4 * pass)) & 15u;
          short* bn = &bins[c % NSET][warp][q * 32 + lane];
          *bn = (short)(*bn + yv[c]);
        }
      }
      int d = 0;
      #pragma unroll
      for (int v = 1; v < 16; ++v) {
        int s = 0;
        #pragma unroll
        for (int st = 0; st < NSET; ++st) { short* bn = &bins[st][warp][v * 32 + lane]; s += *bn; *bn = 0; }
        d += v * s;
      }
      dot[pass] = d;
    }
    int scA, scB, mA, mB; decode_scales(S0, S1, S2, g, scA, scB, mA, mB);
    const float2 dA = dss[buf][sbl * 8 + 2 * g], dB = dss[buf][sbl * 8 + 2 * g + 1];
    facc += finish_unit(dd, dot[0], dot[1], scA, scB, mA, mB, dA, dB);
    if (t + 1 < nt) {
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) *(uint4*)&ys[buf ^ 1][(ii >> 4) * YS32 + (ii & 15) * 4] = yr[i]; }
      if (tid < 64) dss[buf ^ 1][tid] = dr;
      q0 = n0; q1 = n1; S0 = nS0; S1 = nS1; S2 = nS2; dd = ndd;
    }
    __syncthreads();
  }
  facc = warp_sum(facc);
  if (lane == 0) out[row] = facc;
}

#define INSTA(NW) extern "C" __global__ void __launch_bounds__(NW*32) kA_##NW(const uint8_t* W, const u32* y16, const float2* dsf, float* out, int k) { kernA<NW>(W, y16, dsf, out, k); }
#define INSTB(NW) extern "C" __global__ void __launch_bounds__(NW*32) kB_##NW(const uint8_t* W, const int* y32, const float2* dsf, float* out, int k) { kernB<NW>(W, y32, dsf, out, k); }
INSTA(2) INSTA(4) INSTA(8) INSTA(16)
INSTB(2) INSTB(4) INSTB(8)

// ---------------- kernel C: FP32 path (2^23 magic-number nibble -> float, FFMA) ----------------
// f32 activations (raw x). prepf only computes the per-32 sum of x (for folding out the 2^23 bias and the min term).
extern "C" __global__ void prepf(const float* __restrict__ x, int nblk, float2* __restrict__ dsf)
{
  const int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (b >= nblk) return;
  float s = x[b * 32 + lane];
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  if (lane == 0) dsf[b] = make_float2(0.f, s);
}

// FIX=0: exactly as specified (bias folded out per sub-block). FIX=1: subtract 2^23 per weight (FADD) before the FFMA.
template <int NW, int FIX>
static __device__ __forceinline__ void kernC(const uint8_t* __restrict__ W, const float* __restrict__ xf, const float2* __restrict__ dsf, float* __restrict__ out, int k)
{
  constexpr int NT = NW * 32;
  constexpr int NY = (TU * 16 + NT - 1) / NT;
  __shared__ __align__(16) float ys[2][TU * YS32];
  __shared__ float2 dss[2][64];
  const int tid = threadIdx.x, lane = tid & 31;
  const int row = blockIdx.x * NW + (tid >> 5);
  const int nsb = k >> 8, nt = nsb >> 3;
  const uint8_t* rowp = W + (size_t)row * nsb * 144;
  const int sbl = lane >> 2, g = lane & 3;
  const uint4* y4 = (const uint4*)xf;
  uint4 yr[NY]; float2 dr = make_float2(0, 0);
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) yr[i] = y4[ii]; }
  if (tid < 64) dr = dsf[tid];
  #pragma unroll
  for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) *(uint4*)&ys[0][(ii >> 4) * YS32 + (ii & 15) * 4] = yr[i]; }
  if (tid < 64) dss[0][tid] = dr;
  const uint8_t* bp = rowp + sbl * 144;
  uint4 q0 = *(const uint4*)(bp + 16 + 32 * g), q1 = *(const uint4*)(bp + 32 + 32 * g);
  u32 S0 = *(const u32*)(bp + 4), S1 = *(const u32*)(bp + 8), S2 = *(const u32*)(bp + 12), dd = *(const u32*)bp;
  __syncthreads();
  float facc = 0.f;
  for (int t = 0; t < nt; ++t) {
    const int buf = t & 1;
    uint4 n0, n1; u32 nS0, nS1, nS2, ndd;
    if (t + 1 < nt) {
      const uint8_t* np = rowp + ((t + 1) * 8 + sbl) * 144;
      n0 = *(const uint4*)(np + 16 + 32 * g); n1 = *(const uint4*)(np + 32 + 32 * g);
      nS0 = *(const u32*)(np + 4); nS1 = *(const u32*)(np + 8); nS2 = *(const u32*)(np + 12); ndd = *(const u32*)np;
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) yr[i] = y4[(size_t)(t + 1) * TU * 16 + ii]; }
      if (tid < 64) dr = dsf[(t + 1) * 64 + tid];
    }
    const float* yu = &ys[buf][lane * YS32];
    u32 w[8] = { q0.x, q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w };
    float S[2];
    #pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      float a[4] = { 0.f, 0.f, 0.f, 0.f };
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float4 y = *(const float4*)(yu + pass * 32 + 4 * j);
        const float yv[4] = { y.x, y.y, y.z, y.w };
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
          float f = __uint_as_float(((w[j] >> (8 * c + 4 * pass)) & 15u) | 0x4B000000u);
          if (FIX) f -= 8388608.0f;
          a[c] = fmaf(f, yv[c], a[c]);
        }
      }
      S[pass] = (a[0] + a[1]) + (a[2] + a[3]);
    }
    int scA, scB, mA, mB; decode_scales(S0, S1, S2, g, scA, scB, mA, mB);
    const float sxA = dss[buf][sbl * 8 + 2 * g].y, sxB = dss[buf][sbl * 8 + 2 * g + 1].y;
    const float d = __half2float(__ushort_as_half((unsigned short)(dd & 0xffff)));
    const float dmin = __half2float(__ushort_as_half((unsigned short)(dd >> 16)));
    float u;
    if (FIX) u = fmaf(d, fmaf((float)scA, S[0], (float)scB * S[1]), -dmin * fmaf((float)mA, sxA, (float)mB * sxB));
    else {
      const float bA = (float)scA * 8388608.0f, bB = (float)scB * 8388608.0f;
      u = d * fmaf((float)scA, S[0], (float)scB * S[1]) - d * fmaf(bA, sxA, bB * sxB) - dmin * fmaf((float)mA, sxA, (float)mB * sxB);
    }
    facc += u;
    if (t + 1 < nt) {
      #pragma unroll
      for (int i = 0; i < NY; ++i) { int ii = tid + i * NT; if (ii < TU * 16) *(uint4*)&ys[buf ^ 1][(ii >> 4) * YS32 + (ii & 15) * 4] = yr[i]; }
      if (tid < 64) dss[buf ^ 1][tid] = dr;
      q0 = n0; q1 = n1; S0 = nS0; S1 = nS1; S2 = nS2; dd = ndd;
    }
    __syncthreads();
  }
  facc = warp_sum(facc);
  if (lane == 0) out[row] = facc;
}
#define INSTC(NW) \
 extern "C" __global__ void __launch_bounds__(NW*32) kC_##NW(const uint8_t* W, const float* x, const float2* dsf, float* out, int k) { kernC<NW,0>(W, x, dsf, out, k); } \
 extern "C" __global__ void __launch_bounds__(NW*32) kF_##NW(const uint8_t* W, const float* x, const float2* dsf, float* out, int k) { kernC<NW,1>(W, x, dsf, out, k); }
INSTC(2) INSTC(4) INSTC(8) INSTC(16)
