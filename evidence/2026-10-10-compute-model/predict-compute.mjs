#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const FORMATS = {
	q4k: { weights: 256, bytes: 144 },
	q5k: { weights: 256, bytes: 176 },
	q6k: { weights: 256, bytes: 210 },
	q8_0: { weights: 32, bytes: 34 },
	iq1s: { weights: 256, bytes: 50 },
	iq2xxs: { weights: 256, bytes: 66 },
	iq2xs: { weights: 256, bytes: 74 },
	iq3xxs: { weights: 256, bytes: 98 },
	iq4nl: { weights: 32, bytes: 18 },
	f32: { weights: 1, bytes: 4 },
};
const ORDER = ["q4k", "q5k", "q6k", "q8_0", "iq1s", "iq2xxs", "iq2xs", "iq3xxs", "iq4nl", "f32"];
const OPS = new Set(["Contraction", "ExpertIn", "ExpertOut"]);
const SCHEDULERS = 4;
const WORKGROUPS = 16;
const WAVES = 16;
const LANES = 32;
const CLOCK_MHZ = Number(process.env.M60_SM_CLOCK_MHZ || 1174);
const CUOBJDUMP = process.env.CUOBJDUMP || "/opt/cuda/bin/cuobjdump";

function usage() {
	return "Usage: node predict-compute.mjs <trace.trace>... -- <artifact.cubin|artifact.sass.txt>...";
}

function fail(message) {
	throw new Error(message + "\n" + usage());
}

const args = process.argv.slice(2);
const separator = args.indexOf("--");
if (separator < 1) fail("Supply one or more traces before --");
const traces = args.slice(0, separator);
const artifacts = args.slice(separator + 1);
if (!artifacts.length || artifacts.some((file) => !file.endsWith(".cubin") && !file.endsWith(".sass.txt"))) fail("Supply cached cubins or their SASS dumps after --");

function median(values) {
	if (!values.length) return NaN;
	const sorted = [...values].sort((a, b) => a - b);
	const middle = sorted.length >> 1;
	return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function shape(value) {
	const match = /^(\d+)x(\d+)/.exec(value || "");
	return match ? [+match[1], +match[2]] : null;
}

function chain(rows, terms, unitWeights) {
	const rowGroups = Math.ceil(rows / LANES);
	const perWorkgroup = Math.max(1, Math.ceil(rowGroups / WORKGROUPS));
	const teams = Math.min(perWorkgroup, WAVES);
	const teamSize = Math.floor(WAVES / teams);
	const rounds = Math.ceil(perWorkgroup / teams);
	return rounds * Math.ceil(Math.ceil(terms / unitWeights) / teamSize);
}

function sassFunctions(sass) {
	const headers = [...sass.matchAll(/^\s*Function : ([^\r\n]+)/gm)];
	return headers.map((header, index) => ({
		name: header[1].trim(),
		text: sass.slice(header.index, index + 1 < headers.length ? headers[index + 1].index : sass.length),
	}));
}

function loopMetrics(functionText) {
	const instructions = [];
	for (const line of functionText.split(/\r?\n/)) {
		const match = /\/\*([0-9a-f]{4,})\*\/\s+(.*?)\s*;/i.exec(line);
		if (match) instructions.push({ address: Number.parseInt(match[1], 16), text: match[2].trim() });
	}
	const loops = [];
	for (const branch of instructions) {
		const match = /\bBRA\s+(?:CC\.\w+,\s*)?0x([0-9a-f]+)/i.exec(branch.text);
		if (!match) continue;
		const start = Number.parseInt(match[1], 16);
		if (start > branch.address) continue;
		const body = instructions.filter((instruction) => instruction.address >= start && instruction.address <= branch.address);
		const opCounts = new Map();
		for (const instruction of body) {
			const op = instruction.text.replace(/^@!?P\d+\s+/, "").split(/[\s.]/)[0];
			opCounts.set(op, (opCounts.get(op) || 0) + 1);
		}
		const ffma = opCounts.get("FFMA") || 0;
		const fadd = opCounts.get("FADD") || 0;
		if (ffma < 16 && fadd < 128) continue;
		loops.push({
			instructions: body.length,
			i2f: opCounts.get("I2F") || 0,
			ffma,
			fadd,
			start,
			end: branch.address,
		});
	}
	return loops.sort((a, b) => a.instructions - b.instructions || b.ffma - a.ffma);
}

const captures = traces.map((trace) => {
	const lines = fs.readFileSync(trace, "utf8").split(/\r?\n/);
	const metadata = new Map();
	for (const line of lines) {
		const match = /^precision node (\d+) .*?, node \d+ (\w+), input (\d+x\d+), output (\d+x\d+),/.exec(line);
		if (!match) continue;
		const fields = Object.fromEntries([...line.matchAll(/(?:^|\s)(\w+)=([^\s]+)/g)].map((field) => [field[1], field[2]]));
		if (fields.device === undefined) {
			if (["stored_formats", "stored_bytes", "shard_rows", "shard_terms", "local_rows", "local_terms", "run_key", "packed"].some((name) => fields[name] !== undefined)) {
				fail(trace + ": precision node " + match[1] + " is missing device");
			}
			continue;
		}
		if (!OPS.has(match[2])) {
			metadata.set(fields.device + "|" + match[1], { op: match[2], inp: match[3], out: match[4], fields });
			continue;
		}
		for (const name of ["device", "stored_formats", "stored_bytes", "shard_rows", "shard_terms", "local_rows", "local_terms", "run_key", "packed"]) {
			if (fields[name] === undefined) fail(trace + ": precision node " + match[1] + " is missing " + name);
		}
		if (!["true", "false"].includes(fields.packed) || /^\d+$/.exec(fields.stored_bytes) === null) fail(trace + ": invalid storage fields for node " + match[1]);
		const input = shape(match[3]);
		const output = shape(match[4]);
		for (const [axis, channels] of [["rows", output[0]], ["terms", input[0]]]) {
			if (/^\d+:\d+:\d+$/.exec(fields["shard_" + axis]) === null || /^\d+$/.exec(fields["local_" + axis]) === null) fail(trace + ": invalid " + axis + " for node " + match[1]);
			const [first, count, period] = fields["shard_" + axis].split(":").map(Number);
			const local = Number(fields["local_" + axis]);
			const expected = count === 0 ? channels : period === 0 ? count : Math.floor(channels / period) * count;
			if (!Number.isSafeInteger(local) || local <= 0 || local !== expected || (period > 0 && (channels % period || first + count > period))) {
				fail(trace + ": local_" + axis + " disagrees with captured shard for node " + match[1]);
			}
		}
		const key = fields.device + "|" + match[1];
		const row = { op: match[2], inp: match[3], out: match[4], fields };
		if (metadata.has(key) && JSON.stringify(metadata.get(key)) !== JSON.stringify(row)) fail(trace + ": conflicting precision capture for " + key);
		metadata.set(key, row);
	}
	if (![...metadata.values()].some((row) => OPS.has(row.op))) fail(trace + ": no captured per-device compute nodes");
	return { trace, lines, metadata };
});

const kernelVariants = new Map();
const runKeys = new Set(captures.flatMap(({ metadata }) => [...metadata.values()].filter((row) => OPS.has(row.op) && row.fields.packed === "true").flatMap((row) =>
	row.fields.stored_formats.includes("+") ? row.fields.stored_formats.split("+").map((format) => format.split(":")[0]) : [row.fields.run_key])));
for (const artifact of artifacts) {
	if (!fs.existsSync(artifact)) fail("Cached artifact does not exist: " + artifact);
	const sass = artifact.endsWith(".sass.txt") ? fs.readFileSync(artifact, "utf8") : execFileSync(CUOBJDUMP, ["-sass", artifact], { encoding: "utf8", maxBuffer: 512 * 1024 * 1024 });
	const functions = sassFunctions(sass);
	for (const key of runKeys) {
		const preferred = ["packed_lanes_body_$_" + key];
		let candidate = null;
		for (const name of preferred) {
			const matches = functions.filter((item) => item.name === name);
			for (const item of matches) {
				const loops = loopMetrics(item.text);
				if (loops.length && (!candidate || loops[0].instructions > candidate.loop.instructions)) {
					candidate = { name, loop: loops[0] };
				}
			}
			if (candidate) break;
		}
		if (!candidate) continue;
		const variants = kernelVariants.get(key) || kernelVariants.set(key, []).get(key);
		variants.push({ ...candidate, artifact });
	}
}
const kernelMetrics = new Map();
for (const [key, variants] of kernelVariants) {
	const instructionCounts = variants.map((variant) => variant.loop.instructions);
	const i2fCounts = variants.map((variant) => variant.loop.i2f);
	kernelMetrics.set(key, {
		name: variants[0].name,
		loop: {
			instructions: median(instructionCounts),
			instructionMin: Math.min(...instructionCounts),
			instructionMax: Math.max(...instructionCounts),
			i2f: median(i2fCounts),
			i2fMin: Math.min(...i2fCounts),
			i2fMax: Math.max(...i2fCounts),
		},
		variants: variants.length,
	});
}

const nodes = [];
let unmatchedPhases = 0;
let nonPackedPhases = 0;
let mixedNodes = 0;
let unsplittableMixedNodes = 0;
const capturedNodes = captures.reduce((sum, capture) => sum + [...capture.metadata.values()].filter((row) => OPS.has(row.op)).length, 0);
for (const { trace, lines, metadata } of captures) {
	const decodeStarts = [];
	for (const line of lines) {
		const match = /^clocks window (\d+)\.\.(\d+) /.exec(line);
		if (match && +match[2] - +match[1] === 1) decodeStarts.push(+match[1]);
	}
	if (!decodeStarts.length) fail("Trace has no single-token decode window: " + trace);
	const window = Math.min(...decodeStarts);
	const traceName = path.resolve(trace);
	for (const line of lines) {
		const match = /^probe phase (\S+) window (\d+) node (\d+) staged (-?\d+) route (-?\d+) chunk (-?\d+) exit (-?\d+)/.exec(line);
		if (!match || +match[2] !== window) continue;
		const device = match[1];
		const phaseNode = +match[3];
		const row = metadata.get(device + "|" + phaseNode);
		if (!row) {
			unmatchedPhases++;
			fail(trace + ": phase has no per-device precision capture for " + device + "|" + phaseNode);
		}
		if (!OPS.has(row.op) || row.fields.packed !== "true") {
			nonPackedPhases++;
			continue;
		}
		const input = shape(row.inp);
		const output = shape(row.out);
		const computeUs = (+match[6] - +match[4]) / 1000;
		if (!input || !output || computeUs <= 0) {
			unmatchedPhases++;
			continue;
		}
		const localRows = Number(row.fields.local_rows);
		const localTerms = Number(row.fields.local_terms);
		const formats = row.fields.stored_formats.split("+").map((entry) => {
			const match = /^(\w+):(\d+)$/.exec(entry);
			if (!match || !FORMATS[match[1]] || Number(match[2]) <= 0) fail(trace + ": unsupported stored format for " + device + "|" + phaseNode + ": " + entry);
			return [match[1], Number(match[2])];
		});
		if (!formats.length) {
			nonPackedPhases++;
			continue;
		}
		if (formats.length > 1) mixedNodes++;
		const segments = [];
		let rowSum = 0;
		for (const [format, weights] of formats) {
			const spec = FORMATS[format];
			let rowsForFormat = localRows;
			if (formats.length > 1) {
				rowsForFormat = weights / localTerms;
				if (!Number.isSafeInteger(rowsForFormat) || rowsForFormat <= 0) {
					unsplittableMixedNodes++;
					segments.length = 0;
					break;
				}
			}
			rowSum += rowsForFormat;
			const n = chain(rowsForFormat, localTerms, spec.weights);
			const key = formats.length === 1 ? row.fields.run_key : format;
			if (key.replace(/_lanes\d+$/, "") !== format) fail(trace + ": run_key disagrees with stored format for " + device + "|" + phaseNode);
			const metric = kernelMetrics.get(key);
			if (n <= 0) {
				segments.length = 0;
				break;
			}
			segments.push({ format, units: n, baseUs: metric ? SCHEDULERS * metric.loop.instructions * n / CLOCK_MHZ : NaN });
		}
		if (!segments.length || (formats.length > 1 && Math.abs(rowSum - localRows) > 1)) continue;
		nodes.push({
			key: traceName + "|" + device + "|" + phaseNode,
			trace: traceName,
			device,
			node: phaseNode,
			op: row.op,
			formats: segments.map((segment) => segment.format),
			segments,
			computeUs,
			train: false,
		});
	}
}

const groups = new Map();
for (const node of nodes) {
	const key = node.trace + "|" + node.device;
	(groups.get(key) || groups.set(key, []).get(key)).push(node);
}
for (const group of groups.values()) {
	group.sort((a, b) => a.node - b.node);
	let mixedIndex = 0;
	for (const node of group) {
		if (node.formats.length !== 1) {
			node.train = mixedIndex++ % 2 === 0;
			continue;
		}
		const format = node.formats[0];
		const pure = group.filter((entry) => entry.formats.length === 1 && entry.formats[0] === format);
		node.train = pure.indexOf(node) % 2 === 0;
	}
}

const fits = new Map();
const tableRows = [];
for (const format of ORDER) {
	const pure = nodes.filter((node) => node.formats.length === 1 && node.formats[0] === format).sort((a, b) => a.trace.localeCompare(b.trace) || a.device.localeCompare(b.device) || a.node - b.node);
	const train = pure.filter((node) => node.train);
	const held = pure.filter((node) => !node.train);
	const metrics = [...kernelMetrics].filter(([key]) => key.replace(/_lanes\d+$/, "") === format).map(([, metric]) => metric);
	const metric = metrics.length ? {
		loop: {
			instructionMin: Math.min(...metrics.map((metric) => metric.loop.instructionMin)),
			instructionMax: Math.max(...metrics.map((metric) => metric.loop.instructionMax)),
			i2fMin: Math.min(...metrics.map((metric) => metric.loop.i2fMin)),
			i2fMax: Math.max(...metrics.map((metric) => metric.loop.i2fMax)),
		},
	} : null;
	let eta = NaN;
	let medianUnitUs = NaN;
	let error = NaN;
	if (pure.length) {
		medianUnitUs = median(pure.map((node) => node.computeUs / node.segments[0].units));
		if (metric && train.length && held.length) {
			const fitted = train.filter((node) => Number.isFinite(node.segments[0].baseUs));
			const numerator = fitted.reduce((sum, node) => sum + node.segments[0].baseUs ** 2, 0);
			const denominator = fitted.reduce((sum, node) => sum + node.segments[0].baseUs * node.computeUs, 0);
			eta = denominator > 0 ? numerator / denominator : NaN;
			error = median(held.map((node) => Math.abs((node.segments[0].baseUs / eta) / node.computeUs - 1))) * 100;
			if (eta > 0 && eta <= 1.5) fits.set(format, eta);
		}
	}
	tableRows.push({ format, metric, medianUnitUs, eta, error, train: train.length, held: held.length });
}

const heldNodeErrors = [];
let predictedNodes = 0;
let totalSegments = 0;
for (const node of nodes) {
	if (node.train) continue;
	let predictedUs = 0;
	let valid = true;
	for (const segment of node.segments) {
		const eta = fits.get(segment.format);
		if (!eta || !Number.isFinite(segment.baseUs)) {
			valid = false;
			break;
		}
		predictedUs += segment.baseUs / eta;
		totalSegments++;
	}
	if (!valid || predictedUs <= 0) continue;
	predictedNodes++;
	heldNodeErrors.push(Math.abs(predictedUs / node.computeUs - 1) * 100);
}

function display(value, digits = 2) {
	return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}

console.log("# Compute model per format");
console.log("");
console.log("SASS comes from cached cubins or their SASS dumps. Compute time uses the first decode window and phase-clock delta (chunk minus staged). The fit uses alternate nodes per format; the remaining nodes are held out. Rows, terms, and stored formats come from each device's precision trace. The M60 model uses 4 schedulers, 16 workgroups, 16 waves, 32 output lanes, and " + CLOCK_MHZ + " MHz. The fit uses the median SASS count when cached artifacts differ.");
console.log("");
console.log("| Format | Weights per unit | SASS instructions per unit | I2F per unit | Measured us per unit | Fitted eta | Held-out median error | Train/holdout nodes |");
console.log("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |");
for (const row of tableRows) {
	const instructionCount = row.metric ? row.metric.loop.instructionMin === row.metric.loop.instructionMax ? String(row.metric.loop.instructionMin) : row.metric.loop.instructionMin + "-" + row.metric.loop.instructionMax : "n/a";
	const i2f = row.metric ? row.metric.loop.i2fMin === row.metric.loop.i2fMax ? String(row.metric.loop.i2fMin) : row.metric.loop.i2fMin + "-" + row.metric.loop.i2fMax : "n/a";
	const weights = String(FORMATS[row.format].weights);
	const measured = display(row.medianUnitUs, 4);
	const eta = display(row.eta, 3);
	const error = display(row.error, 1) + (Number.isFinite(row.error) ? "%" : "");
	console.log("| " + row.format + " | " + weights + " | " + instructionCount + " | " + i2f + " | " + measured + " | " + eta + " | " + error + " | " + row.train + "/" + row.held + " |");
}
console.log("");
const allError = median(heldNodeErrors);
console.log("Supported held-out nodes: " + predictedNodes + "/" + nodes.filter((node) => !node.train).length + "; median absolute percentage error: " + display(allError, 1) + "%; threshold: 25%.");
console.log("Phase nodes without a matching shape: " + unmatchedPhases + "; non-packed phases: " + nonPackedPhases + "; mixed-format nodes: " + mixedNodes + "; mixed nodes without a valid row split: " + unsplittableMixedNodes + ".");
console.log("Captured per-device compute nodes: " + capturedNodes + ".");
console.log("Cached artifact variants: " + [...kernelMetrics].map(([format, metric]) => format + "=" + metric.variants).join(", ") + ".");
if (!Number.isFinite(allError) || allError > 25 || unmatchedPhases > 0) process.exitCode = 1;
