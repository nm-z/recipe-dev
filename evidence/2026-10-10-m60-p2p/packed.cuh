// Packed Q4_K XMAD helper from #1091, commit 8118be62cb9ac7189439b0e7b38663d764914a10.
#include <cuda_fp16.h>
// Unit = (super-block, group g): 32 bytes of qs = sub-blocks 2g (low nibbles) and 2g+1 (high nibbles).
// One warp = one output row; lane l of tile t handles super-block 8t+(l>>2), group (l&3).
#include <stdint.h>
typedef unsigned int u32;
#define TU 32
#define M2 0x000F000Fu
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
