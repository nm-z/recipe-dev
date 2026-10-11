/* Reader and writer for the RCPREF01 layout written by recipe reference.rs. */
#ifndef RCPREF_H
#define RCPREF_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define RCPREF_STEPS_OFFSET 16

typedef struct {
	uint32_t vocab;
	uint64_t steps;
} rcpref_header;

static inline void rcpref_die(const char *msg)
{
	fprintf(stderr, "error: %s\n", msg);
	exit(1);
}

/* Records use native memcpy on little-endian machines. */
static inline void rcpref_require_little_endian(void)
{
	uint32_t one = 1;
	unsigned char first;
	memcpy(&first, &one, 1);
	if (first != 1)
		rcpref_die("big-endian machines are not supported");
}

static inline void rcpref_write_header(FILE *f, uint32_t vocab, uint64_t steps)
{
	const uint32_t version = 1;
	if (fwrite("RCPREF01", 1, 8, f) != 8 || fwrite(&version, 4, 1, f) != 1 || fwrite(&vocab, 4, 1, f) != 1 || fwrite(&steps, 8, 1, f) != 1)
		rcpref_die("cannot write RCPREF01 header");
}

static inline void rcpref_patch_steps(FILE *f, uint64_t steps)
{
	if (fseek(f, RCPREF_STEPS_OFFSET, SEEK_SET) != 0 || fwrite(&steps, 8, 1, f) != 1)
		rcpref_die("cannot patch RCPREF01 step count");
}

static inline rcpref_header rcpref_read_header(FILE *f)
{
	unsigned char magic[8];
	uint32_t version;
	rcpref_header h;
	if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "RCPREF01", 8) != 0)
		rcpref_die("not an RCPREF01 file");
	if (fread(&version, 4, 1, f) != 1 || version != 1)
		rcpref_die("unsupported RCPREF01 version");
	if (fread(&h.vocab, 4, 1, f) != 1 || fread(&h.steps, 8, 1, f) != 1)
		rcpref_die("short RCPREF01 header");
	if (h.vocab == 0)
		rcpref_die("RCPREF01 vocabulary is empty");
	if (h.steps == 0)
		rcpref_die("RCPREF01 has no steps (writer did not close)");
	return h;
}

static inline void rcpref_write_record(FILE *f, uint64_t index, uint32_t winner, uint32_t vocab, const double *logits)
{
	if (fwrite(&index, 8, 1, f) != 1 || fwrite(&winner, 4, 1, f) != 1 || fwrite(logits, 8, vocab, f) != vocab)
		rcpref_die("cannot write RCPREF01 step");
}

static inline void rcpref_read_record(FILE *f, uint32_t vocab, uint64_t *index, uint32_t *winner, double *logits)
{
	if (fread(index, 8, 1, f) != 1 || fread(winner, 4, 1, f) != 1 || fread(logits, 8, vocab, f) != vocab)
		rcpref_die("short RCPREF01 step");
}

static inline void rcpref_skip_record(FILE *f, uint32_t vocab, uint64_t *index, uint32_t *winner)
{
	if (fread(index, 8, 1, f) != 1 || fread(winner, 4, 1, f) != 1)
		rcpref_die("short RCPREF01 step");
	if (fseek(f, (long)(8 * (size_t)vocab), SEEK_CUR) != 0)
		rcpref_die("cannot skip RCPREF01 logits");
}

#endif
