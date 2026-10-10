#include <stdint.h>

// Both allocations are owned by their receiving die and peer-mapped by the sender.
struct Packet {
	uint32_t value[4096];
	uint32_t sequence;
	uint32_t pad[15];
	uint32_t done;
};
struct Report {
	uint64_t start_ns, end_ns;
	uint32_t errors, timeout, sent, received;
	float max_abs, max_rel;
};
static __device__ __forceinline__ uint64_t timer_ns() {
	uint64_t t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}
static __device__ __forceinline__ uint32_t read_word(const uint32_t* p) {
	uint32_t v;
	asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
	return v;
}
static __device__ __forceinline__ void write_word(uint32_t* p, uint32_t v) {
	asm volatile("st.volatile.global.u32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
static __device__ __forceinline__ uint4 read_quad(const uint32_t* p) {
	uint4 v;
	asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
		: "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
	return v;
}
static __device__ __forceinline__ void write_quad(uint32_t* p, uint4 v) {
	asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
		:: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
static __device__ __forceinline__ bool wait_word(const uint32_t* p, uint32_t want, uint64_t deadline) {
	while (read_word(p) != want) if (timer_ns() >= deadline) return false;
	return true;
}
static __device__ __forceinline__ void publish(Packet* remote, uint32_t sequence) {
	// Fence every writing thread. A fence in thread 0 alone does not order other threads' writes.
	__threadfence_system();
	__syncthreads();
	if (threadIdx.x == 0) {
		write_word(&remote->sequence, sequence);
		__threadfence_system();
	}
	__syncthreads();
}
extern "C" __global__ void relay(Packet* local, Packet* remote, Report* report, uint32_t die,
	uint32_t hops, uint32_t base, uint32_t seed, uint32_t delay_ns, uint32_t skip_flag) {
	__shared__ uint32_t ok, errors[256];
	__shared__ uint64_t start, deadline;
	uint32_t bad = 0;
	const uint32_t tid = threadIdx.x;
	if (die == 0) for (uint32_t i = tid; i < 4096; i += 256) local->value[i] = seed + i;
	__syncthreads();
	if (tid == 0) {
		start = timer_ns();
		deadline = start + 1000000000ULL;
		report->sent = report->received = report->timeout = 0;
	}
	__syncthreads();
	// Retain each consumed vector in registers for the next publication.
	uint4 payload[4];
	if (die == 0) {
		#pragma unroll
		for (uint32_t tile = 0; tile < 4; ++tile) payload[tile] = read_quad(&local->value[(tid + tile * 256) * 4]);
	}
	for (uint32_t hop = 1; hop <= hops; ++hop) {
		const uint32_t sender = (hop - 1) & 1;
		if (die == sender) {
			#pragma unroll
			for (uint32_t tile = 0; tile < 4; ++tile) write_quad(&remote->value[(tid + tile * 256) * 4], payload[tile]);
			if (delay_ns && tid == 0) {
				uint64_t until = timer_ns() + delay_ns;
				while (timer_ns() < until) {}
			}
			if (!(skip_flag && hop == 1)) publish(remote, base + hop);
			if (tid == 0) ++report->sent;
		} else {
			if (tid == 0) ok = wait_word(&local->sequence, base + hop, deadline);
			__syncthreads();
			if (!ok) { if (tid == 0) report->timeout = hop; break; }
			#pragma unroll
			for (uint32_t tile = 0; tile < 4; ++tile) {
				uint32_t i = (tid + tile * 256) * 4;
				uint4 v = read_quad(&local->value[i]);
				uint32_t want = seed + i + hop - 1;
				bad += (v.x != want) + (v.y != want + 1) + (v.z != want + 2) + (v.w != want + 3);
				payload[tile] = make_uint4(v.x + 1, v.y + 1, v.z + 1, v.w + 1);
			}
			__syncthreads();
			if (tid == 0) ++report->received;
		}
	}
	// For odd hop counts, include the last consumer's validation and return acknowledgement.
	if (hops & 1) {
		if (die == 1 && !report->timeout) {
			__threadfence_system();
			__syncthreads();
			if (tid == 0) { write_word(&remote->done, base + hops); __threadfence_system(); }
		} else if (die == 0 && tid == 0 && !wait_word(&local->done, base + hops, deadline)) report->timeout = hops + 1;
	}
	errors[tid] = bad;
	__syncthreads();
	for (uint32_t stride = 128; stride; stride >>= 1) {
		if (tid < stride) errors[tid] += errors[tid + stride];
		__syncthreads();
	}
	if (tid == 0) {
		report->start_ns = start;
		report->end_ns = timer_ns();
		report->errors = errors[0];
	}
}

#include "packed.cuh"
// A persistent CTA reuses the selected packed helper over all 4096 rows.
// This demonstrates device-only layer handoff, not full-die matvec throughput.
static __device__ void prepare(Packet* packet, u32* y16, float2* dsf) {
	int lane = threadIdx.x & 31;
	for (int tile = 0; tile < 16; ++tile) {
		int b = tile * 8 + (threadIdx.x >> 5);
		float v = __uint_as_float(read_word(&packet->value[b * 32 + lane]));
		float amax = fabsf(v);
		for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
		float d = amax / 127.f;
		int q = amax == 0.f ? 0 : (int)roundf(v / d);
		int sum = q;
		for (int o = 16; o; o >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
		int partner = __shfl_xor_sync(0xffffffff, q, 2);
		int p = lane >> 2, c = lane & 3;
		if (c < 2) y16[b * 16 + 2 * p + (c & 1)] = (q & 65535) | ((partner & 65535) << 16);
		if (lane == 0) { float d8 = __half2float(__float2half(d)); dsf[b] = make_float2(d8, d8 * sum); }
	}
	__syncthreads();
}
extern "C" __global__ void layer(Packet* local, Packet* remote, Report* report, u32 die, u32 base,
	const uint8_t* weights, u32* y16, float2* dsf, float* output, const float* expected) {
	__shared__ u32 ok, errors[256];
	__shared__ float max_abs[256], max_rel[256];
	__shared__ uint64_t start, deadline;
	int tid = threadIdx.x;
	if (tid == 0) {
		start = timer_ns(); deadline = start + 1000000000ULL;
		report->sent = report->received = report->timeout = 0;
		ok = 1;
		if (die == 1) ok = wait_word(&local->sequence, base + 1, deadline);
	}
	__syncthreads();
	if (!ok) { if (tid == 0) report->timeout = 1; return; }
	prepare(local, y16, dsf);
	for (int tile = 0; tile < 512; ++tile) {
		kernA<8>(weights + (size_t)tile * 8 * 16 * 144, y16, dsf, output + tile * 8, 4096);
		__syncthreads();
	}
	u32 bad = 0;
	float abs_max = 0.f, rel_max = 0.f;
	for (int i = tid; i < 4096; i += 256) {
		float delta = fabsf(output[i] - expected[i]);
		abs_max = fmaxf(abs_max, delta);
		rel_max = fmaxf(rel_max, delta / fmaxf(fabsf(expected[i]), 1.e-6f));
		bad += !isfinite(output[i]) || delta > 1.e-4f + 1.e-3f * fabsf(expected[i]);
		if (die == 0) write_word(&remote->value[i], __float_as_uint(output[i]));
	}
	__syncthreads();
	if (die == 0) {
		publish(remote, base + 1);
		if (tid == 0) {
			report->sent = 1;
			if (!wait_word(&local->done, base + 1, deadline)) report->timeout = 2;
		}
	} else {
		__threadfence_system();
		__syncthreads();
		if (tid == 0) { write_word(&remote->done, base + 1); __threadfence_system(); report->received = 1; }
	}
	errors[tid] = bad; max_abs[tid] = abs_max; max_rel[tid] = rel_max;
	__syncthreads();
	for (int s = 128; s; s >>= 1) {
		if (tid < s) { errors[tid] += errors[tid+s]; max_abs[tid] = fmaxf(max_abs[tid], max_abs[tid+s]); max_rel[tid] = fmaxf(max_rel[tid], max_rel[tid+s]); }
		__syncthreads();
	}
	if (tid == 0) {
		report->start_ns = start; report->end_ns = timer_ns(); report->errors = errors[0];
		report->max_abs = max_abs[0]; report->max_rel = max_rel[0];
	}
}
