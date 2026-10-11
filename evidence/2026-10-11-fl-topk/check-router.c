#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct { float weights[512]; uint32_t selected[11]; } Routing;
typedef struct { float value; uint32_t id; } Candidate;
static float bits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t word(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float router_exp(float x) {
	if (x < bits(0xc2cff1b5)) return 0.0f;
	float scaled = x * bits(0x3fb8aa3b);
	float shifted = scaled + 0.5f;
	float kf = floorf(shifted);
	int32_t k = (int32_t)kf;
	float upper = kf * bits(0xbf317218);
	float partial = x + upper;
	float lower = kf * bits(0xaf51cf7a);
	float r = partial + lower;
	float p = fmaf(r, bits(0x37d00d01), bits(0x39500d01));
	p = fmaf(p, r, bits(0x3ab60b61));
	p = fmaf(p, r, bits(0x3c088889));
	p = fmaf(p, r, bits(0x3d2aaaab));
	p = fmaf(p, r, bits(0x3e2aaaab));
	p = fmaf(p, r, 0.5f);
	p = fmaf(p, r, 1.0f);
	float body = r * p;
	float expanded = body + 1.0f;
	int32_t half = k >= 0 ? k / 2 : -((-k + 1) / 2);
	int32_t first = half < -127 ? -127 : half;
	first += 127;
	if (first > 254) first = 254;
	int32_t second = k - half + 127;
	if (second < 0) second = 0;
	if (second > 254) second = 254;
	float value = expanded * bits((uint32_t)first << 23);
	return value * bits((uint32_t)second << 23);
}
static Candidate better(Candidate a, Candidate b) {
	return b.value > a.value || (b.value == a.value && b.id < a.id) ? b : a;
}
static void scalar(const float scores[512], Routing *out) {
	uint32_t ids[10], count = 0;
	memset(out, 0, sizeof(*out));
	for (uint32_t id = 0; id < 512; id++) {
		if (count == 10 && !(scores[id] > scores[ids[9]])) continue;
		uint32_t j = count < 10 ? count : 9;
		while (j && scores[id] > scores[ids[j - 1]]) { ids[j] = ids[j - 1]; j--; }
		ids[j] = id;
		if (count < 10) count++;
	}
	for (uint32_t i = 0; i < 10; i++) out->weights[ids[i]] = 1.0f;
	float maximum = 0.0f;
	int first = 1;
	for (uint32_t id = 0; id < 512; id++) if (out->weights[id] == 1.0f) {
		if (first || scores[id] > maximum) maximum = scores[id];
		first = 0;
	}
	float sum = 0.0f;
	for (uint32_t id = 0; id < 512; id++) {
		float value = out->weights[id] == 1.0f ? router_exp(scores[id] - maximum) : 0.0f;
		out->weights[id] = value;
		sum = sum + value;
	}
	for (uint32_t id = 0; id < 512; id++) {
		float raw = out->weights[id];
		out->weights[id] = raw == 0.0f ? 0.0f : raw / sum;
		if (out->weights[id] != 0.0f) out->selected[++out->selected[0]] = id;
	}
}
static int cta(const float scores[512], uint32_t width, Routing *out) {
	for (uint32_t id = 0; id < 512; id++) if (!isfinite(scores[id])) return 0;
	Candidate lanes[512], warps[32], winners[10];
	float current[512];
	memcpy(current, scores, sizeof(current));
	memset(out, 0, sizeof(*out));
	for (uint32_t pick = 0; pick < 10; pick++) {
		for (uint32_t lane = 0; lane < width; lane++) {
			Candidate a = { current[lane], current[lane] == -INFINITY ? UINT32_MAX : lane };
			Candidate b = { -INFINITY, UINT32_MAX };
			if (lane + width < 512) { b.value = current[lane + width]; b.id = b.value == -INFINITY ? UINT32_MAX : lane + width; }
			lanes[lane] = better(a, b);
		}
		for (uint32_t step = 16; step; step >>= 1)
			for (uint32_t lane = 0; lane < width; lane++)
				if ((lane & 31) + step < 32) lanes[lane] = better(lanes[lane], lanes[lane + step]);
		for (uint32_t lane = 0; lane < 32; lane++) {
			Candidate invalid = { -INFINITY, UINT32_MAX };
			warps[lane] = lane < width / 32 ? lanes[lane * 32] : invalid;
		}
		for (uint32_t step = 16; step; step >>= 1)
			for (uint32_t lane = 0; lane + step < 32; lane++) warps[lane] = better(warps[lane], warps[lane + step]);
		winners[pick] = warps[0];
		current[winners[pick].id] = -INFINITY;
	}
	float maximum = winners[0].value;
	for (uint32_t i = 1; i < 10; i++) {
		Candidate value = winners[i];
		uint32_t j = i;
		while (j && winners[j - 1].id > value.id) { winners[j] = winners[j - 1]; j--; }
		winners[j] = value;
	}
	float weights[10], sum = 0.0f;
	for (uint32_t i = 0; i < 10; i++) { weights[i] = router_exp(winners[i].value - maximum); sum = sum + weights[i]; }
	for (uint32_t i = 0; i < 10; i++) {
		float value = weights[i] / sum;
		out->weights[winners[i].id] = value;
		if (value != 0.0f) out->selected[++out->selected[0]] = winners[i].id;
	}
	return 1;
}
static uint32_t seed = 0x10925120;
static uint32_t random_word(void) { seed = seed * 1664525u + 1013904223u; return seed; }
static int weighted_return_check(void) {
	static float planes[5 * 10 * 2560], coefficients[5 * 10];
	static float compact[5 * 2560], old_return[5 * 4096], fused_return[5 * 4096];
	uint32_t counts[5];
	uint64_t comparisons = 0;
	for (uint32_t case_id = 0; case_id < 55; case_id++) {
		uint32_t positions = case_id / 11 + 1;
		for (uint32_t position = 0; position < positions; position++) {
			counts[position] = (case_id + position) % 11;
			for (uint32_t slot = 0; slot < 10; slot++) {
				coefficients[position * 10 + slot] = (float)(random_word() & 0xffffu) / 65536.0f;
				for (uint32_t row = 0; row < 2560; row++)
					planes[(position * 10 + slot) * 2560 + row] = slot < counts[position] ? (float)(int32_t)random_word() / 16777216.0f : NAN;
			}
			for (uint32_t row = 0; row < 2560; row++) {
				float sum = 0.0f;
				for (uint32_t slot = 0; slot < counts[position]; slot++)
					sum = fmaf(coefficients[position * 10 + slot], planes[(position * 10 + slot) * 2560 + row], sum);
				compact[position * 2560 + row] = sum;
			}
			for (uint32_t row = 0; row < 4096; row++) old_return[position * 4096 + row] = row < 2560 ? compact[position * 2560 + row] : 0.0f;
		}
		for (uint32_t lane = 0; lane < 4096; lane++) for (uint32_t linear = lane; linear < positions * 4096; linear += 4096) {
			uint32_t position = linear / 4096, row = linear % 4096;
			float sum = 0.0f;
			if (row < 2560) for (uint32_t slot = 0; slot < counts[position]; slot++)
				sum = fmaf(coefficients[position * 10 + slot], planes[(position * 10 + slot) * 2560 + row], sum);
			fused_return[linear] = sum;
		}
		for (uint32_t linear = 0; linear < positions * 4096; linear++) {
			comparisons++;
			if (word(old_return[linear]) != word(fused_return[linear])) { fprintf(stderr, "Weighted return differs at case %u, index %u.\n", case_id, linear); return 1; }
		}
	}
	printf("Weighted-return CPU comparisons: %llu output words across active counts 0..10 and positions 1..5; zero mismatches.\n", (unsigned long long)comparisons);
	return 0;
}
int main(void) {
	float scores[512];
	Routing reference, candidate;
	uint64_t comparisons = 0;
	for (uint32_t case_id = 0; case_id < 4108; case_id++) {
		for (uint32_t id = 0; id < 512; id++) {
			uint32_t random = random_word();
			switch (case_id) {
				case 0: scores[id] = 1.0f; break;
				case 1: scores[id] = (float)id; break;
				case 2: scores[id] = -(float)id; break;
				case 3: scores[id] = id & 1 ? -0.0f : 0.0f; break;
				case 4: scores[id] = (float)(id % 13); break;
				case 5: scores[id] = -(float)(id % 13); break;
				case 6: scores[id] = id < 9 ? 1.0f : 0.0f; break;
				case 7: scores[id] = id < 11 ? 1.0f : 0.0f; break;
				case 8: scores[id] = id < 7 ? 0.0f : -104.0f; break;
				case 9: scores[id] = id == 511 ? FLT_MAX : -FLT_MAX; break;
				case 10: scores[id] = bits((random & 0x80000000u) | (random & 0x007fffffu)); break;
				case 11: scores[id] = bits(0x7f7fffffu - id); break;
				default: scores[id] = (float)(int32_t)random / 16777216.0f; break;
			}
		}
		scalar(scores, &reference);
		for (uint32_t width = 256; width <= 512; width *= 2) {
			if (!cta(scores, width, &candidate)) return 1;
			if (memcmp(reference.selected, candidate.selected, sizeof(reference.selected))) { fprintf(stderr, "Selected IDs differ at case %u, width %u.\n", case_id, width); return 1; }
			for (uint32_t id = 0; id < 512; id++) {
				comparisons++;
				if (word(reference.weights[id]) != word(candidate.weights[id])) { fprintf(stderr, "Coefficient differs at case %u, width %u, expert %u.\n", case_id, width, id); return 1; }
			}
		}
	}
	for (uint32_t kind = 0; kind < 3; kind++) for (uint32_t id = 0; id < 512; id++) {
		memset(scores, 0, sizeof(scores));
		scores[id] = kind == 0 ? INFINITY : kind == 1 ? -INFINITY : NAN;
		if (cta(scores, 256, &candidate) || cta(scores, 512, &candidate)) return 1;
	}
	printf("Router CPU comparisons: %llu coefficients, 8216 selected lists, 3072 nonfinite fallbacks; zero mismatches.\n", (unsigned long long)comparisons);
	return weighted_return_check();
}
