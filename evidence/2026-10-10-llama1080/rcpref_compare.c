/*
	* rcpref-compare: streams a reference and a candidate RCPREF01 file and measures each logit's
	* absolute error against the reference's logit range at the same position.
	*
	*	rcpref-compare <reference.rcpref> <candidate.rcpref> [--tsv <file>] [--limit <fraction>] [--bin <steps>] [--max-slope <points per 1000 steps>]
	*/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "rcpref.h"

typedef struct {
	double sum_ratio;
	double max_ratio;
	double sum_abs;
	uint64_t n;
} bin_t;

static void usage(void)
{
	fprintf(stderr, "usage: rcpref-compare <reference.rcpref> <candidate.rcpref> [--tsv <file>] [--limit <fraction>] [--bin <steps>] [--max-slope <points per 1000 steps>]\n");
	exit(2);
}

/* Least-squares slope of y against x = 0..n-1. */
static double slope(const double *y, uint64_t n)
{
	double mx = (double)(n - 1) / 2.0, my = 0;
	for (uint64_t i = 0; i < n; i++)
		my += y[i];
	my /= (double)n;
	double sxy = 0, sxx = 0;
	for (uint64_t i = 0; i < n; i++) {
		double dx = (double)i - mx;
		sxy += dx * (y[i] - my);
		sxx += dx * dx;
	}
	return sxx > 0 ? sxy / sxx : 0;
}

static int compare_double(const void *left, const void *right)
{
	double a = *(const double *)left, b = *(const double *)right;
	return (a > b) - (a < b);
}

int main(int argc, char **argv)
{
	rcpref_require_little_endian();
	if (argc < 3)
		usage();
	const char *ref_path = argv[1];
	const char *cand_path = argv[2];
	const char *tsv_path = NULL;
	double limit = 0.10;
	double max_slope = NAN;
	uint64_t bin_size = 256;
	for (int i = 3; i < argc; i++) {
		if (strcmp(argv[i], "--tsv") == 0 && i + 1 < argc)
			tsv_path = argv[++i];
		else if (strcmp(argv[i], "--limit") == 0 && i + 1 < argc)
			limit = atof(argv[++i]);
		else if (strcmp(argv[i], "--bin") == 0 && i + 1 < argc)
			bin_size = strtoull(argv[++i], NULL, 10);
		else if (strcmp(argv[i], "--max-slope") == 0 && i + 1 < argc) {
			char *end;
			max_slope = strtod(argv[++i], &end);
			if (end == argv[i] || *end != '\0' || !isfinite(max_slope))
				usage();
		}
		else
			usage();
	}
	if (bin_size == 0)
		rcpref_die("bin size must be positive");

	FILE *fa = fopen(ref_path, "rb");
	FILE *fb = fopen(cand_path, "rb");
	if (!fa || !fb)
		rcpref_die("cannot open inputs");
	rcpref_header ha = rcpref_read_header(fa);
	rcpref_header hb = rcpref_read_header(fb);
	if (ha.vocab != hb.vocab)
		rcpref_die("vocabulary sizes differ");
	if (ha.steps != hb.steps)
		rcpref_die("step counts differ");
	uint32_t V = ha.vocab;
	uint64_t N = ha.steps;

	double *a = malloc(sizeof *a * V);
	double *b = malloc(sizeof *b * V);
	double *ratio = malloc(sizeof *ratio * N);
	double *maxabs = malloc(sizeof *maxabs * N);
	uint64_t nbins = (N + bin_size - 1) / bin_size;
	bin_t *bins = calloc(nbins, sizeof *bins);
	FILE *tsv = NULL;
	if (tsv_path) {
		tsv = fopen(tsv_path, "w");
		if (!tsv)
			rcpref_die("cannot open TSV output");
		fprintf(tsv, "step\trange\tmax_abs_diff\targmax_logit\tratio\twinner_match\n");
	}

	uint64_t winner_mismatch = 0;
	double global_ratio = 0;
	uint64_t global_step = 0, global_logit = 0;
	for (uint64_t k = 0; k < N; k++) {
		uint64_t ia, ib;
		uint32_t wa, wb;
		rcpref_read_record(fa, V, &ia, &wa, a);
		rcpref_read_record(fb, V, &ib, &wb, b);
		if (ia != k || ib != k)
			rcpref_die("step index out of order");
		if (!isfinite(a[0]) || !isfinite(b[0]))
			rcpref_die("nonfinite logit");
		uint32_t ar = 0, br = 0;
		for (uint32_t j = 1; j < V; j++) {
			if (!isfinite(a[j]) || !isfinite(b[j]))
				rcpref_die("nonfinite logit");
			if (a[j] > a[ar])
				ar = j;
			if (b[j] > b[br])
				br = j;
		}
		int match = ar == br;
		if (!match)
			winner_mismatch++;

		double lo = a[0], hi = a[0];
		for (uint32_t j = 1; j < V; j++) {
			if (a[j] < lo)
				lo = a[j];
			if (a[j] > hi)
				hi = a[j];
		}
		double range = hi - lo;

		double mx = 0;
		uint32_t arg = 0;
		for (uint32_t j = 0; j < V; j++) {
			double d = fabs(a[j] - b[j]);
			if (isnan(d))
				d = INFINITY;
			if (d > mx) {
				mx = d;
				arg = j;
			}
		}
		double r = range > 0 ? mx / range : (mx > 0 ? INFINITY : 0);
		ratio[k] = r;
		maxabs[k] = mx;
		if (r > global_ratio) {
			global_ratio = r;
			global_step = k;
			global_logit = arg;
		}

		bin_t *bn = &bins[k / bin_size];
		bn->sum_ratio += r;
		bn->sum_abs += mx;
		bn->n++;
		if (r > bn->max_ratio)
			bn->max_ratio = r;

		if (tsv)
			fprintf(tsv, "%llu\t%.9g\t%.9g\t%u\t%.9g\t%d\n", (unsigned long long)k, range, mx, arg, r, match);
	}
	if (tsv)
		fclose(tsv);

	printf("steps %llu vocab %u winner_mismatches %llu\n", (unsigned long long)N, V, (unsigned long long)winner_mismatch);
	printf("global max |diff| / reference range %.4f%% at step %llu logit %llu\n", 100.0 * global_ratio, (unsigned long long)global_step, (unsigned long long)global_logit);
	printf("bins of %llu steps: first_step last_step mean_ratio_pct p99_ratio_pct max_ratio_pct mean_max_abs_diff\n", (unsigned long long)bin_size);
	for (uint64_t i = 0; i < nbins; i++) {
		bin_t *bn = &bins[i];
		uint64_t first = i * bin_size;
		uint64_t last = first + bn->n - 1;
		double *sorted = malloc(sizeof *sorted * bn->n);
		if (!sorted)
			rcpref_die("cannot allocate bin quantiles");
		memcpy(sorted, ratio + first, sizeof *sorted * bn->n);
		qsort(sorted, bn->n, sizeof *sorted, compare_double);
		uint64_t p99 = (uint64_t)ceil(0.99 * (double)bn->n) - 1;
		printf("bin %llu %llu %.4f %.4f %.4f %.6g\n", (unsigned long long)first, (unsigned long long)last, 100.0 * bn->sum_ratio / (double)bn->n, 100.0 * sorted[p99], 100.0 * bn->max_ratio, bn->sum_abs / (double)bn->n);
		free(sorted);
	}
	printf("slope max_abs_diff per 1000 steps %.6g (logit units)\n", 1000.0 * slope(maxabs, N));
	double ratio_slope = 100.0 * 1000.0 * slope(ratio, N);
	printf("slope ratio per 1000 steps %.6g (percentage points of range)\n", ratio_slope);

	int pass = winner_mismatch == 0 && global_ratio <= limit;
	if (!isnan(max_slope)) {
		printf("bound check %s (limit %.4f%% of reference range, winners %s)\n", pass ? "PASS" : "FAIL", 100.0 * limit, winner_mismatch == 0 ? "identical" : "differ");
		int flat = isfinite(ratio_slope) && ratio_slope <= max_slope;
		printf("growth check %s (slope %.6g, maximum %.6g percentage points per 1000 steps)\n", flat ? "PASS" : "FAIL", ratio_slope, max_slope);
		pass = pass && flat;
	}
	printf("verdict %s (limit %.4f%% of reference range, winners %s)\n", pass ? "PASS" : "FAIL", 100.0 * limit, winner_mismatch == 0 ? "identical" : "differ");

	free(a);
	free(b);
	free(ratio);
	free(maxabs);
	free(bins);
	fclose(fa);
	fclose(fb);
	return pass ? 0 : 1;
}
