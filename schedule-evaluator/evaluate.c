#define _POSIX_C_SOURCE 200809L

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
	DEVICES = 2,
	ROWS = 5,
	OPERAND_BYTES = 40,
	RESULT_BYTES = 20,
	TOTAL_BYTES = OPERAND_BYTES + RESULT_BYTES,
	MAX_TRANSFER = 40,
};

static const double SPEED[DEVICES] = {40.0, 80.0};
static const double RATE[DEVICES] = {2.0, 10.0};
static const double LINK = 60.0;

typedef struct {
	bool active;
	int rows[ROWS];
	int count;
	int completed;
	double start;
	double finish;
} Calculation;

typedef struct {
	bool active;
	int to;
	int bytes[MAX_TRANSFER];
	int count;
	int completed;
	double start;
	double finish;
} Transfer;

typedef struct {
	double time;
	bool resident[DEVICES][TOTAL_BYTES + 1];
	bool processed[ROWS];
	bool reserved[ROWS];
	Calculation calculation[DEVICES];
	Transfer transfer[DEVICES];
	int calculation_bytes[DEVICES];
	int transfer_bytes[DEVICES];
	int processed_bytes;
} State;

typedef struct {
	int work[DEVICES][ROWS];
	int work_count[DEVICES];
	int bytes[MAX_TRANSFER];
	int byte_count;
	int from;
	int to;
} Plan;

static bool fail(const char **error, const char *message) {
	if (error != NULL) {
		*error = message;
	}
	return false;
}

static void state_new(State *state) {
	memset(state, 0, sizeof(*state));
	for (int byte = 1; byte <= OPERAND_BYTES; ++byte) {
		state->resident[0][byte] = true;
	}
}

static int count_bytes(const bool bytes[TOTAL_BYTES + 1], int first, int last) {
	int count = 0;
	for (int byte = first; byte <= last; ++byte) {
		count += bytes[byte] ? 1 : 0;
	}
	return count;
}

static bool active(const State *state) {
	for (int device = 0; device < DEVICES; ++device) {
		if (state->calculation[device].active || state->transfer[device].active) {
			return true;
		}
	}
	return false;
}

static bool done(const State *state) {
	return state->processed_bytes == RESULT_BYTES
		&& count_bytes(state->resident[0], OPERAND_BYTES + 1, TOTAL_BYTES) == RESULT_BYTES
		&& !active(state);
}

static bool ready(const State *state, int device, int row) {
	return !state->processed[row]
		&& !state->reserved[row]
		&& count_bytes(state->resident[device], row * 8 + 1, row * 8 + 8) == 8;
}

static bool integer_between(double value, int low, int high) {
	return isfinite(value) && value >= low && value <= high && floor(value) == value;
}

static bool plan(const State *state, double ti_value, double tf_value, double tb_value,
		double cb_value, double mask_value, Plan *result, const char **error) {
	memset(result, 0, sizeof(*result));
	if (!((ti_value == 1.0 || ti_value == 2.0) && (tf_value == 1.0 || tf_value == 2.0))) {
		return fail(error, "devices must be 1 or 2");
	}
	if (!integer_between(tb_value, 4, 40)) {
		return fail(error, "tb must be integer bytes in [4,40]");
	}
	if (!integer_between(cb_value, 4, 40)) {
		return fail(error, "cb must be integer bytes in [4,40]");
	}
	if (!integer_between(mask_value, 0, 3)) {
		return fail(error, "c must be 0,1,2,3");
	}

	const int ti = (int)ti_value - 1;
	const int tf = (int)tf_value - 1;
	const int tb = (int)tb_value;
	const int cb = (int)cb_value;
	const int mask = (int)mask_value;
	if (mask != 0 && cb % 8 != 0) {
		return fail(error, "cb must contain complete 8-byte pairs");
	}
	if (done(state)) {
		return fail(error, "work is already complete");
	}

	const bool selected[DEVICES] = {
		mask == 1 || mask == 3,
		mask == 2 || mask == 3,
	};
	for (int device = 0; device < DEVICES; ++device) {
		if (selected[device] && state->calculation[device].active) {
			return fail(error, "calculation device is busy");
		}
	}

	bool taken[ROWS] = {false};
	const int order[DEVICES] = {1, 0};
	for (int index = 0; index < DEVICES; ++index) {
		const int device = order[index];
		if (!selected[device]) {
			continue;
		}
		for (int row = 0; row < ROWS; ++row) {
			if (result->work_count[device] < cb / 8 && !taken[row] && ready(state, device, row)) {
				result->work[device][result->work_count[device]++] = row;
				taken[row] = true;
			}
		}
		if (result->work_count[device] != cb / 8) {
			return fail(error, "calculation device lacks cb bytes of distinct ready pairs");
		}
	}

	result->from = ti;
	result->to = tf;
	if (ti != tf) {
		if (state->transfer[ti].active) {
			return fail(error, "transfer direction is busy");
		}
		for (int pass = 0; pass < 2; ++pass) {
			const int first = pass == 0 ? OPERAND_BYTES + 1 : 1;
			const int last = pass == 0 ? TOTAL_BYTES : OPERAND_BYTES;
			for (int byte = first; byte <= last && result->byte_count < tb; ++byte) {
				if (state->resident[ti][byte] && !state->resident[tf][byte]) {
					result->bytes[result->byte_count++] = byte;
				}
			}
		}
		if (result->byte_count != tb) {
			return fail(error, "source lacks enough ready bytes absent from destination");
		}
	}

	if (result->byte_count == 0 && result->work_count[0] == 0 && result->work_count[1] == 0
			&& !active(state)) {
		return fail(error, "decision makes no progress");
	}
	return true;
}

static bool advance(State *state, const char **error) {
	double time = INFINITY;
	for (int device = 0; device < DEVICES; ++device) {
		if (state->calculation[device].active && state->calculation[device].finish < time) {
			time = state->calculation[device].finish;
		}
		if (state->transfer[device].active && state->transfer[device].finish < time) {
			time = state->transfer[device].finish;
		}
	}
	if (!isfinite(time)) {
		return fail(error, "no pending completion");
	}

	for (int device = 0; device < DEVICES; ++device) {
		Calculation *calculation = &state->calculation[device];
		if (calculation->active) {
			int completed = (int)floor((time - calculation->start) * RATE[device] + 1e-9);
			if (completed > calculation->count) {
				completed = calculation->count;
			}
			for (int index = calculation->completed; index < completed; ++index) {
				const int row = calculation->rows[index];
				if (state->processed[row]) {
					return fail(error, "duplicate calculation");
				}
				for (int byte = OPERAND_BYTES + 1 + row * 4;
						byte <= OPERAND_BYTES + 4 + row * 4; ++byte) {
					state->resident[device][byte] = true;
				}
				state->processed[row] = true;
				state->reserved[row] = false;
				state->calculation_bytes[device] += 4;
				state->processed_bytes += 4;
			}
			calculation->completed = completed;
			if (calculation->finish <= time) {
				calculation->active = false;
			}
		}

		Transfer *transfer = &state->transfer[device];
		if (transfer->active) {
			int completed = (int)floor((time - transfer->start) * LINK + 1e-9);
			if (completed > transfer->count) {
				completed = transfer->count;
			}
			for (int index = transfer->completed; index < completed; ++index) {
				state->resident[transfer->to][transfer->bytes[index]] = true;
			}
			state->transfer_bytes[device] += completed - transfer->completed;
			transfer->completed = completed;
			if (transfer->finish <= time) {
				transfer->active = false;
			}
		}
	}

	state->time = time;
	if (state->processed_bytes > RESULT_BYTES
			|| count_bytes(state->resident[0], OPERAND_BYTES + 1, TOTAL_BYTES) > state->processed_bytes) {
		return fail(error, "result accounting failed");
	}
	return true;
}

static bool step(State *state, double ti, double tf, double tb, double cb, double mask,
		const char **error) {
	Plan next;
	if (!plan(state, ti, tf, tb, cb, mask, &next, error)) {
		return false;
	}
	for (int device = 0; device < DEVICES; ++device) {
		if (next.work_count[device] == 0) {
			continue;
		}
		Calculation *calculation = &state->calculation[device];
		calculation->active = true;
		calculation->count = next.work_count[device];
		calculation->completed = 0;
		calculation->start = state->time;
		calculation->finish = state->time + next.work_count[device] / RATE[device];
		for (int index = 0; index < next.work_count[device]; ++index) {
			calculation->rows[index] = next.work[device][index];
			state->reserved[next.work[device][index]] = true;
		}
	}
	if (next.byte_count > 0) {
		Transfer *transfer = &state->transfer[next.from];
		transfer->active = true;
		transfer->to = next.to;
		transfer->count = next.byte_count;
		transfer->completed = 0;
		transfer->start = state->time;
		transfer->finish = state->time + next.byte_count / LINK;
		memcpy(transfer->bytes, next.bytes, (size_t)next.byte_count * sizeof(next.bytes[0]));
	}
	return advance(state, error);
}

static double remaining_calculation(const State *state, int device) {
	if (!state->calculation[device].active) {
		return 0.0;
	}
	return fmax(0.0, state->calculation[device].finish - state->time);
}

static double remaining_transfer(const State *state, int device) {
	if (!state->transfer[device].active) {
		return 0.0;
	}
	return fmax(0.0, state->transfer[device].finish - state->time);
}

static void snapshot(const State *state, double values[16]) {
	values[0] = SPEED[0];
	values[1] = RATE[0];
	values[2] = SPEED[1];
	values[3] = RATE[1];
	values[4] = count_bytes(state->resident[0], 1, TOTAL_BYTES);
	values[5] = count_bytes(state->resident[1], 1, TOTAL_BYTES);
	values[6] = state->processed_bytes;
	values[7] = count_bytes(state->resident[0], OPERAND_BYTES + 1, TOTAL_BYTES);
	values[8] = state->calculation_bytes[0];
	values[9] = remaining_calculation(state, 0);
	values[10] = state->calculation_bytes[1];
	values[11] = remaining_calculation(state, 1);
	values[12] = state->transfer_bytes[0];
	values[13] = remaining_transfer(state, 0);
	values[14] = state->transfer_bytes[1];
	values[15] = remaining_transfer(state, 1);
}

static bool print_state(const State *state, FILE *stream, const char **error) {
	static const char *const names[16] = {
		"d1s", "d1r", "d2s", "d2r", "r1", "r2", "p", "e",
		"c1b", "c1t", "c2b", "c2t", "t1b", "t1t", "t2b", "t2t",
	};
	double values[16];
	snapshot(state, values);
	for (int index = 0; index < 16; ++index) {
		if (fprintf(stream, "%s%s=%.12g", index == 0 ? "" : ",", names[index], values[index]) < 0) {
			return fail(error, "cannot write evaluator state");
		}
	}
	if (fputc('\n', stream) == EOF || fflush(stream) == EOF) {
		return fail(error, "cannot write evaluator state");
	}
	return true;
}

static bool print_choices(const State *state, FILE *stream, bool protocol, size_t *count,
		const char **error) {
	*count = 0;
	for (int ti = 1; ti <= 2; ++ti) {
		for (int tf = 1; tf <= 2; ++tf) {
			for (int tb = 4; tb <= 40; ++tb) {
				for (int cb = 4; cb <= 40; ++cb) {
					for (int mask = 0; mask <= 3; ++mask) {
						Plan candidate;
						if (!plan(state, ti, tf, tb, cb, mask, &candidate, NULL)) {
							continue;
						}
						if (fprintf(stream, "%s%d,%d,%d,%d,%d\n", protocol ? "choice " : "",
								ti, tf, tb, cb, mask) < 0) {
							return fail(error, "cannot write evaluator choices");
						}
						++*count;
					}
				}
			}
		}
	}
	return true;
}

static bool frame(const State *state, const char **error) {
	if (done(state)) {
		if (printf("score %.17g\n", state->time) < 0 || fflush(stdout) == EOF) {
			return fail(error, "cannot write evaluator score");
		}
		return true;
	}
	if (fputs("state ", stdout) == EOF || !print_state(state, stdout, error)) {
		return false;
	}
	if (puts("actions ti,tf,tb,cb,c") == EOF) {
		return fail(error, "cannot write evaluator action schema");
	}
	size_t choices = 0;
	if (!print_choices(state, stdout, true, &choices, error)) {
		return false;
	}
	(void)choices;
	if (puts("ready") == EOF || fflush(stdout) == EOF) {
		return fail(error, "cannot write evaluator frame");
	}
	return true;
}

static bool parse_number(const char *text, double *value) {
	char *end = NULL;
	*value = strtod(text, &end);
	if (end == text || !isfinite(*value)) {
		return false;
	}
	while (isspace((unsigned char)*end)) {
		++end;
	}
	return *end == '\0';
}

static bool parse_action(char *record, double values[5], const char **error) {
	static const char *const names[5] = {"ti", "tf", "tb", "cb", "c"};
	char *seen[5] = {NULL};
	int fields = 0;
	char *save = NULL;
	for (char *field = strtok_r(record, ",", &save); field != NULL;
			field = strtok_r(NULL, ",", &save)) {
		char *equals = strchr(field, '=');
		if (equals == NULL || equals == field || equals[1] == '\0') {
			return fail(error, "invalid or duplicate action field");
		}
		*equals = '\0';
		for (int previous = 0; previous < fields; ++previous) {
			if (strcmp(field, seen[previous]) == 0) {
				return fail(error, "invalid or duplicate action field");
			}
		}
		if (fields == 5) {
			return fail(error, "expected ti,tf,tb,cb,c");
		}
		seen[fields] = field;
		int index = -1;
		for (int candidate = 0; candidate < 5; ++candidate) {
			if (strcmp(field, names[candidate]) == 0) {
				index = candidate;
				break;
			}
		}
		double value;
		if (!parse_number(equals + 1, &value)) {
			return fail(error, "action must be numeric");
		}
		if (index >= 0) {
			values[index] = value;
		}
		++fields;
	}
	if (fields != 5) {
		return fail(error, "expected ti,tf,tb,cb,c");
	}
	return true;
}

static void remove_newline(char *line) {
	const size_t length = strlen(line);
	if (length > 0 && line[length - 1] == '\n') {
		line[length - 1] = '\0';
	}
}

static bool protocol(const char **error) {
	State state;
	bool initialized = false;
	char *line = NULL;
	size_t capacity = 0;
	ssize_t length;
	while ((length = getline(&line, &capacity, stdin)) >= 0) {
		(void)length;
		remove_newline(line);
		if (strcmp(line, "close") == 0) {
			if (!initialized || !done(&state)) {
				free(line);
				return fail(error, "cannot close an unfinished episode");
			}
			free(line);
			return true;
		}
		if (strcmp(line, "reset") == 0) {
			if (initialized && !done(&state)) {
				free(line);
				return fail(error, "cannot reset an unfinished episode");
			}
			state_new(&state);
			initialized = true;
			if (!frame(&state, error)) {
				free(line);
				return false;
			}
			continue;
		}
		if (!initialized || strncmp(line, "choose ", 7) != 0 || line[7] == '\0') {
			free(line);
			return fail(error, "expected reset or choose");
		}
		double values[5] = {NAN, NAN, NAN, NAN, NAN};
		if (!parse_action(line + 7, values, error)
				|| !step(&state, values[0], values[1], values[2], values[3], values[4], error)
				|| !frame(&state, error)) {
			free(line);
			return false;
		}
	}
	const bool read_failed = ferror(stdin);
	free(line);
	if (read_failed) {
		return fail(error, "cannot read evaluator input");
	}
	if (!initialized || !done(&state)) {
		return fail(error, "evaluator input closed before episode completed");
	}
	return true;
}

static bool blank(const char *line) {
	for (; *line != '\0'; ++line) {
		if (!isspace((unsigned char)*line)) {
			return false;
		}
	}
	return true;
}

static bool choices_command(char *line) {
	while (isspace((unsigned char)*line)) {
		++line;
	}
	char *end = line + strlen(line);
	while (end > line && isspace((unsigned char)end[-1])) {
		--end;
	}
	*end = '\0';
	return strcmp(line, "choices") == 0;
}

static bool parse_schedule(char *line, double values[5], const char **error) {
	int count = 0;
	char *cursor = line;
	while (*cursor != '\0') {
		while (*cursor != '\0' && (*cursor == ',' || isspace((unsigned char)*cursor))) {
			++cursor;
		}
		if (*cursor == '\0') {
			break;
		}
		char *end = NULL;
		const double value = strtod(cursor, &end);
		if (end == cursor || !isfinite(value)) {
			return fail(error, "expected numeric ti tf tb cb c");
		}
		if (*end != '\0' && *end != ',' && !isspace((unsigned char)*end)) {
			return fail(error, "expected numeric ti tf tb cb c");
		}
		if (count < 5) {
			values[count] = value;
		}
		++count;
		cursor = end;
	}
	if (count != 5) {
		return fail(error, "expected ti tf tb cb c");
	}
	return true;
}

static bool direct(bool interactive, bool trace, const char **error) {
	State state;
	state_new(&state);
	if (interactive && !print_state(&state, stdout, error)) {
		return false;
	}
	if (trace && !print_state(&state, stderr, error)) {
		return false;
	}

	char *line = NULL;
	size_t capacity = 0;
	ssize_t length;
	while ((length = getline(&line, &capacity, stdin)) >= 0) {
		(void)length;
		remove_newline(line);
		char *comment = strchr(line, '#');
		if (comment != NULL) {
			*comment = '\0';
		}
		if (blank(line)) {
			continue;
		}
		if (interactive && choices_command(line)) {
			size_t choices = 0;
			if (!print_choices(&state, stdout, false, &choices, error)
					|| puts("end") == EOF || fflush(stdout) == EOF) {
				free(line);
				return fail(error, "cannot write evaluator choices");
			}
			continue;
		}
		double values[5] = {0.0};
		if (!parse_schedule(line, values, error)
				|| !step(&state, values[0], values[1], values[2], values[3], values[4], error)) {
			free(line);
			return false;
		}
		if (interactive && !print_state(&state, stdout, error)) {
			free(line);
			return false;
		}
		if (trace && !print_state(&state, stderr, error)) {
			free(line);
			return false;
		}
	}
	const bool read_failed = ferror(stdin);
	free(line);
	if (read_failed) {
		return fail(error, "cannot read evaluator input");
	}
	if (interactive) {
		return true;
	}
	while (active(&state)) {
		if (!advance(&state, error)) {
			return false;
		}
	}
	if (!done(&state)) {
		return fail(error, "incomplete schedule: all 20 result bytes must reach d1");
	}
	if (printf("%.12g\n", state.time) < 0) {
		return fail(error, "cannot write evaluator score");
	}
	return true;
}

static bool close_enough(double left, double right) {
	return fabs(left - right) < 1e-8;
}

static bool check(bool condition, const char **error) {
	return condition || fail(error, "self-check failed");
}

static bool self_check(const char **error) {
	State state;
	double values[16];
	const char *ignored = NULL;

	state_new(&state);
	if (!check(!step(&state, 1, 1, 4, 4, 1, &ignored), error)
			|| !step(&state, 1, 1, 4, 8, 1, error)) {
		return false;
	}
	snapshot(&state, values);
	if (!check(state.processed_bytes == 4 && values[4] == 44 && !done(&state)
			&& close_enough(state.time, 1.0 / RATE[0]), error)) {
		return false;
	}
	for (int iteration = 0; iteration < 4; ++iteration) {
		if (!step(&state, 1, 1, 4, 8, 1, error)) {
			return false;
		}
	}
	if (!check(done(&state) && close_enough(state.time, 5.0 / RATE[0]), error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 1, 4, 40, 1, error)) {
		return false;
	}
	snapshot(&state, values);
	if (!check(done(&state) && close_enough(state.time, 5.0 / RATE[0])
			&& values[4] == 60 && state.processed_bytes == 20, error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 2, 40, 4, 0, error)
			|| !step(&state, 2, 2, 4, 40, 2, error)
			|| !step(&state, 2, 1, 20, 4, 0, error)) {
		return false;
	}
	snapshot(&state, values);
	if (!check(done(&state) && close_enough(state.time, 40.0 / LINK + 5.0 / RATE[1] + 20.0 / LINK)
			&& values[4] == 60 && values[5] == 60, error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 2, 4, 4, 0, error)) {
		return false;
	}
	snapshot(&state, values);
	ignored = NULL;
	if (!check(state.transfer_bytes[0] == 4 && values[5] == 4 && state.processed_bytes == 0
			&& !step(&state, 2, 2, 4, 8, 2, &ignored)
			&& close_enough(state.time, 4.0 / LINK), error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 2, 16, 4, 0, error)) {
		return false;
	}
	ignored = NULL;
	if (!check(!step(&state, 1, 1, 4, 24, 3, &ignored), error)
			|| !step(&state, 1, 1, 4, 8, 3, error)) {
		return false;
	}
	if (!check(state.calculation[0].active && !state.calculation[1].active
			&& state.calculation_bytes[1] == 4 && state.processed_bytes == 4, error)
			|| !step(&state, 2, 1, 4, 4, 0, error)) {
		return false;
	}
	while (active(&state)) {
		if (!advance(&state, error)) {
			return false;
		}
	}
	if (!check(state.calculation_bytes[0] == 4 && state.calculation_bytes[1] == 4
			&& state.processed_bytes == 8, error)
			|| !step(&state, 1, 1, 4, 24, 1, error)) {
		return false;
	}
	while (active(&state)) {
		if (!advance(&state, error)) {
			return false;
		}
	}
	if (!check(done(&state) && close_enough(state.time, 16.0 / LINK + 4.0 / RATE[0]), error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 2, 16, 4, 0, error)
			|| !step(&state, 1, 2, 16, 16, 3, error)) {
		return false;
	}
	if (!check(state.calculation_bytes[1] == 8 && state.calculation[0].active
			&& state.transfer[0].active, error)
			|| !step(&state, 2, 1, 8, 4, 0, error)) {
		return false;
	}
	if (!check(!state.transfer[0].active && state.transfer[1].active
			&& state.transfer_bytes[0] == 32 && state.transfer_bytes[1] > 0
			&& state.transfer_bytes[1] < 8, error)) {
		return false;
	}
	while (active(&state)) {
		if (!advance(&state, error)) {
			return false;
		}
	}
	if (!check(state.calculation_bytes[0] == 8 && state.calculation_bytes[1] == 8, error)
			|| !step(&state, 1, 1, 4, 8, 1, error)) {
		return false;
	}
	while (active(&state)) {
		if (!advance(&state, error)) {
			return false;
		}
	}
	if (!check(done(&state) && close_enough(state.time, 16.0 / LINK + 3.0 / RATE[0]), error)) {
		return false;
	}

	state_new(&state);
	if (!step(&state, 1, 2, 40, 40, 1, error)) {
		return false;
	}
	if (!check(state.calculation[0].active && !state.transfer[0].active
			&& state.processed_bytes < 20 && !done(&state), error)
			|| !advance(&state, error)
			|| !check(done(&state) && close_enough(state.time, 5.0 / RATE[0]), error)) {
		return false;
	}

	state_new(&state);
	ignored = NULL;
	if (!check(!step(&state, 1, 1, 4, 8, 2, &ignored), error)) {
		return false;
	}
	ignored = NULL;
	if (!check(!step(&state, 2, 1, 4, 4, 0, &ignored), error)) {
		return false;
	}
	ignored = NULL;
	if (!check(!step(&state, 1, 1, 41, 8, 1, &ignored), error)) {
		return false;
	}
	ignored = NULL;
	if (!check(!step(&state, 1, 1, 4, 8, 4, &ignored), error)) {
		return false;
	}
	for (int ti = 1; ti <= 2; ++ti) {
		for (int tf = 1; tf <= 2; ++tf) {
			for (int tb = 4; tb <= 40; ++tb) {
				for (int cb = 4; cb <= 40; ++cb) {
					for (int mask = 0; mask <= 3; ++mask) {
						Plan candidate;
						if (plan(&state, ti, tf, tb, cb, mask, &candidate, NULL)) {
							State copy = state;
							if (!step(&copy, ti, tf, tb, cb, mask, error)) {
								return false;
							}
						}
					}
				}
			}
		}
	}
	if (puts("checks passed") == EOF) {
		return fail(error, "cannot write self-check result");
	}
	return true;
}

int main(int argc, char **argv) {
	const char *error = NULL;
	bool ok;
	const char *protocol_mode = getenv("RECIPE_RAT_PROTOCOL");
	if (protocol_mode != NULL && strcmp(protocol_mode, "1") == 0) {
		ok = protocol(&error);
	} else if (argc > 1 && strcmp(argv[1], "--self-check") == 0) {
		ok = self_check(&error);
	} else {
		const bool interactive = argc > 1 && strcmp(argv[1], "--step") == 0;
		const bool trace = argc > 1 && strcmp(argv[1], "--trace") == 0;
		if (argc > 1 && !interactive && !trace) {
			ok = fail(&error, "use --step, --trace, or --self-check");
		} else {
			ok = direct(interactive, trace, &error);
		}
	}
	if (!ok) {
		fprintf(stderr, "evaluate.c failed: %s\n", error == NULL ? "evaluation failed" : error);
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}
