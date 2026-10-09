//! A recorded reference of a decode and the comparison of another run against it.
//!
//! `RECIPE_REFERENCE_WRITE=<path>` records every step's full logits; `RECIPE_REFERENCE=<path>`
//! reads such a file and checks each step of the current run against it under the
//! profile's tolerance. The two are exclusive. A comparison hands back the reference's
//! winning token at every step so the caller can feed it forward and keep later steps
//! comparable after a flip. A record hands back its own winner for the same reason, so
//! the recorded run walks the greedy path whatever the sampler would have done.
//!
//! File: the magic `RCPREF01`, a u32 format version, a u32 vocabulary, a u64 step count
//! written at close, then per step a u64 step index, a u32 winner, and the vocabulary's
//! f64 logits as little-endian bits. Bits are preserved, so an `exact` comparison is a
//! bit comparison. A record always starts a fresh file; nothing is appended to an old run.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};

const MAGIC: &[u8; 8] = b"RCPREF01";
const VERSION: u32 = 1;
const HEADER: u64 = 8 + 4 + 4 + 8;

/// The outcome of one compared step.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
	/// Within tolerance and the same winner.
	Pass,
	/// Within tolerance, a different winner, and the reference scored its own winner
	/// and the actual winner within the tolerance of each other: reported, not failed.
	Flip { reference: usize, actual: usize, gap: f64 },
	/// Over the tolerance, or a different winner whose gap exceeds it.
	Fail(String),
}

/// What a finished comparison amounts to.
#[derive(Clone, Debug, Default)]
pub struct Summary {
	pub steps: usize,
	pub worst: f64,
	pub flips: Vec<(usize, usize, usize, f64)>,
	pub failures: Vec<(usize, String)>,
}

impl Summary {
	pub fn ok(&self) -> bool {
		self.failures.is_empty()
	}
}

enum Mode {
	Off,
	Record { file: BufWriter<File>, count: u64 },
	Compare { winners: Vec<u32>, logits: Vec<f64>, next: usize },
}

pub struct Reference {
	mode: Mode,
	vocabulary: usize,
	tolerance: f64,
	exact: bool,
	summary: Summary,
}

fn read_u32(bytes: &[u8], at: usize) -> Result<u32, String> {
	bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(|| "reference file is truncated".to_owned())
}

fn read_u64(bytes: &[u8], at: usize) -> Result<u64, String> {
	bytes.get(at..at + 8).map(|b| u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])).ok_or_else(|| "reference file is truncated".to_owned())
}

fn winner(logits: &[f64]) -> usize {
	let mut best = 0;
	for (index, value) in logits.iter().enumerate() {
		if *value > logits[best] {
			best = index;
		}
	}
	best
}

impl Reference {
	/// A reference that neither records nor compares.
	pub fn off() -> Self {
		Self { mode: Mode::Off, vocabulary: 0, tolerance: 0.0, exact: false, summary: Summary::default() }
	}

	/// Opens from the environment. `tolerance` is the profile's absolute logit
	/// tolerance; `exact` asks for a bit comparison and ignores the tolerance.
	pub fn open(tolerance: f64, exact: bool) -> Result<Self, String> {
		let write = std::env::var("RECIPE_REFERENCE_WRITE").ok().filter(|path| !path.is_empty());
		let read = std::env::var("RECIPE_REFERENCE").ok().filter(|path| !path.is_empty());
		if write.is_some() && read.is_some() {
			return Err("RECIPE_REFERENCE and RECIPE_REFERENCE_WRITE are both set; a run records or compares, not both".to_owned());
		}
		if !(tolerance.is_finite() && tolerance >= 0.0) {
			return Err(format!("reference tolerance {tolerance} is not a finite non-negative number"));
		}
		let mode = match (write, read) {
			(Some(path), None) => {
				let file = File::create(&path).map_err(|error| format!("cannot create reference {path}: {error}"))?;
				let mut file = BufWriter::new(file);
				file.write_all(MAGIC).and_then(|()| file.write_all(&VERSION.to_le_bytes())).and_then(|()| file.write_all(&0u32.to_le_bytes())).and_then(|()| file.write_all(&0u64.to_le_bytes())).map_err(|error| format!("cannot write reference {path}: {error}"))?;
				Mode::Record { file, count: 0 }
			}
			(None, Some(path)) => Self::load(&path)?,
			_ => Mode::Off,
		};
		Ok(Self { mode, vocabulary: 0, tolerance, exact, summary: Summary::default() })
	}

	fn load(path: &str) -> Result<Mode, String> {
		let mut bytes = Vec::new();
		File::open(path).and_then(|mut file| file.read_to_end(&mut bytes)).map_err(|error| format!("cannot read reference {path}: {error}"))?;
		if bytes.len() < HEADER as usize || &bytes[..8] != MAGIC {
			return Err(format!("{path} is not a recipe reference file"));
		}
		let version = read_u32(&bytes, 8)?;
		if version != VERSION {
			return Err(format!("{path} is reference format version {version}; this build reads {VERSION}"));
		}
		let vocabulary = read_u32(&bytes, 12)? as usize;
		let count = usize::try_from(read_u64(&bytes, 16)?).map_err(|_| format!("{path} claims more steps than this machine can address"))?;
		if vocabulary == 0 || count == 0 {
			return Err(format!("{path} records no steps"));
		}
		// The header is untrusted: every size is checked before it sizes anything.
		let stride = vocabulary.checked_mul(8).and_then(|bytes| bytes.checked_add(12)).ok_or_else(|| format!("{path} names a vocabulary too large to lay out"))?;
		let expected = count.checked_mul(stride).and_then(|bytes| bytes.checked_add(HEADER as usize)).ok_or_else(|| format!("{path} names more steps than can be laid out"))?;
		if bytes.len() != expected {
			return Err(format!("{path} holds {} bytes; {count} steps of {vocabulary} logits need {expected}", bytes.len()));
		}
		let values = count.checked_mul(vocabulary).ok_or_else(|| format!("{path} names more logits than can be counted"))?;
		let mut winners = Vec::with_capacity(count);
		let mut logits = Vec::with_capacity(values);
		for step in 0..count {
			let at = HEADER as usize + step * stride;
			let index = read_u64(&bytes, at)? as usize;
			if index != step {
				return Err(format!("{path} step {step} is labelled {index}"));
			}
			let recorded = read_u32(&bytes, at + 8)? as usize;
			let base = at + 12;
			for k in 0..vocabulary {
				let b = &bytes[base + k * 8..base + k * 8 + 8];
				let value = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
				if !value.is_finite() {
					return Err(format!("{path} step {step} logit {k} is {value}"));
				}
				logits.push(value);
			}
			let row = &logits[step * vocabulary..(step + 1) * vocabulary];
			if recorded >= vocabulary || winner(row) != recorded {
				return Err(format!("{path} step {step} records winner {recorded}, its logits say {}", winner(row)));
			}
			winners.push(recorded as u32);
		}
		let mode = Mode::Compare { winners, logits, next: 0 };
		Ok(mode)
	}

	/// Records or compares one step. The winner to feed forward comes back: the
	/// reference's in a comparison, this step's own in a record; `None` when off.
	pub fn step(&mut self, step: usize, logits: &[f64]) -> Result<Option<usize>, String> {
		if logits.is_empty() {
			return Err(format!("step {step} has no logits"));
		}
		if let Some(bad) = logits.iter().position(|value| !value.is_finite()) {
			return Err(format!("step {step} logit {bad} is {}", logits[bad]));
		}
		match &mut self.mode {
			Mode::Off => Ok(None),
			Mode::Record { file, count } => {
				if *count == 0 {
					self.vocabulary = logits.len();
				} else if logits.len() != self.vocabulary {
					return Err(format!("step {step} has {} logits; the record began with {}", logits.len(), self.vocabulary));
				}
				if step as u64 != *count {
					return Err(format!("step {step} recorded out of order; expected {count}"));
				}
				let best = winner(logits);
				let mut row = Vec::with_capacity(12 + logits.len() * 8);
				row.extend_from_slice(&(step as u64).to_le_bytes());
				row.extend_from_slice(&(best as u32).to_le_bytes());
				for value in logits {
					row.extend_from_slice(&value.to_bits().to_le_bytes());
				}
				file.write_all(&row).map_err(|error| format!("cannot write reference step {step}: {error}"))?;
				*count += 1;
				// A record hands back its own winner so the recorded run walks the greedy path
				// a comparison will walk, whatever the sampler's temperature.
				Ok(Some(best))
			}
			Mode::Compare { winners, logits: recorded, next } => {
				let vocabulary = recorded.len() / winners.len();
				if step != *next {
					return Err(format!("step {step} compared out of order; expected {next}"));
				}
				if *next >= winners.len() {
					return Err(format!("step {step} is past the reference's {} steps", winners.len()));
				}
				if logits.len() != vocabulary {
					return Err(format!("step {step} has {} logits; the reference has {vocabulary}", logits.len()));
				}
				let expected = &recorded[*next * vocabulary..(*next + 1) * vocabulary];
				let reference = winners[*next] as usize;
				let actual = winner(logits);
				let mut worst = 0.0f64;
				let mut first_bit_mismatch = None;
				for (k, (a, b)) in expected.iter().zip(logits).enumerate() {
					let delta = (a - b).abs();
					if delta > worst {
						worst = delta;
					}
					if first_bit_mismatch.is_none() && a.to_bits() != b.to_bits() {
						first_bit_mismatch = Some(k);
					}
				}
				if worst > self.summary.worst {
					self.summary.worst = worst;
				}
				let verdict = if self.exact {
					match first_bit_mismatch {
						None => Verdict::Pass,
						Some(k) => Verdict::Fail(format!("logit {k} differs: reference {:e}, actual {:e} (exact profile)", expected[k], logits[k])),
					}
				} else if worst > self.tolerance {
					Verdict::Fail(format!("max |delta| {worst:e} exceeds tolerance {:e}", self.tolerance))
				} else if actual != reference {
					let gap = expected[reference] - expected[actual];
					if gap <= self.tolerance { Verdict::Flip { reference, actual, gap } } else { Verdict::Fail(format!("winner {actual} instead of {reference}; the reference scores them {gap:e} apart, over tolerance {:e}", self.tolerance)) }
				} else {
					Verdict::Pass
				};
				match verdict {
					Verdict::Pass => {}
					Verdict::Flip { reference, actual, gap } => self.summary.flips.push((step, reference, actual, gap)),
					Verdict::Fail(why) => self.summary.failures.push((step, why)),
				}
				self.summary.steps += 1;
				*next += 1;
				Ok(Some(reference))
			}
		}
	}

	/// Closes a record (writing the step count) or ends a comparison, checking that every
	/// reference step was reached, and prints one summary line to stderr. The summary's
	/// `ok` is false when any step failed.
	pub fn finish(self) -> Result<Summary, String> {
		let mut summary = self.summary;
		match self.mode {
			Mode::Off => Ok(summary),
			Mode::Record { mut file, count } => {
				if count == 0 {
					return Err("reference record ends with no steps".to_owned());
				}
				file.flush().map_err(|error| format!("cannot flush reference: {error}"))?;
				let mut inner = file.into_inner().map_err(|error| format!("cannot close reference: {error}"))?;
				inner.seek(SeekFrom::Start(12)).and_then(|_| inner.write_all(&(self.vocabulary as u32).to_le_bytes())).and_then(|()| inner.write_all(&count.to_le_bytes())).and_then(|()| inner.flush()).map_err(|error| format!("cannot finalize reference: {error}"))?;
				eprintln!("reference: recorded {count} steps of {} logits", self.vocabulary);
				summary.steps = count as usize;
				Ok(summary)
			}
			Mode::Compare { winners, next, .. } => {
				if next < winners.len() {
					summary.failures.push((next, format!("run ended at step {next}; the reference has {}", winners.len())));
				}
				let flips = if summary.flips.is_empty() { String::new() } else { format!(" at {}", summary.flips.iter().map(|(step, r, a, gap)| format!("{step}({r}->{a} gap {gap:.2e})")).collect::<Vec<_>>().join(" ")) };
				let mode = if self.exact { "exact".to_owned() } else { format!("tolerance {:e}", self.tolerance) };
				eprintln!("reference: {} steps compared, {mode}, max |delta| {:e}, {} flips reported{flips}, {} failures", summary.steps, summary.worst, summary.flips.len(), summary.failures.len());
				for (step, why) in &summary.failures {
					eprintln!("reference: step {step}: {why}");
				}
				Ok(summary)
			}
		}
	}
}

/// What a teacher-forced scoring pass does with the environment's reference: nothing,
/// record every scored position's logits, or compare them against a recorded run.
pub enum Scored {
	Off,
	Record(Reference),
	Compare(String),
}

impl Scored {
	pub fn open(tolerance: f64, exact: bool) -> Result<Self, String> {
		let read = std::env::var("RECIPE_REFERENCE").ok().filter(|path| !path.is_empty());
		match read {
			Some(path) if std::env::var("RECIPE_REFERENCE_WRITE").is_ok_and(|write| !write.is_empty()) => Err(format!("RECIPE_REFERENCE={path} and RECIPE_REFERENCE_WRITE are both set; a run records or compares, not both")),
			Some(path) => Ok(Self::Compare(path)),
			None => {
				let reference = Reference::open(tolerance, exact)?;
				Ok(if matches!(reference.mode, Mode::Record { .. }) { Self::Record(reference) } else { Self::Off })
			}
		}
	}
}

/// How a metric spreads over the compared positions.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spread {
	pub mean: f64,
	pub median: f64,
	pub p99: f64,
	pub min: f64,
	pub max: f64,
}

impl Spread {
	fn of(values: &[f64]) -> Self {
		if values.is_empty() {
			return Self::default();
		}
		let mut sorted = values.to_vec();
		sorted.sort_by(f64::total_cmp);
		let count = sorted.len();
		let median = if count % 2 == 1 { sorted[count / 2] } else { (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0 };
		Self { mean: values.iter().sum::<f64>() / count as f64, median, p99: sorted[(count * 99).div_ceil(100) - 1], min: sorted[0], max: sorted[count - 1] }
	}
}

/// One position's variant logits against the reference's.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionDelta {
	/// KL divergence of the variant's next-token distribution from the reference's, in nats.
	pub kl: f64,
	/// Pearson correlation of the two logit vectors.
	pub pearson: f64,
	/// Spearman rank correlation over the reference's top tokens.
	pub spearman: f64,
	pub max_abs: f64,
	/// `max_abs` over the reference's logit range at this position.
	pub max_relative: f64,
	pub top1: bool,
	pub top5: bool,
	pub reference_logprob: f64,
	pub logprob: f64,
}

/// The rows of the per-position table: positions `first..=last`.
#[derive(Clone, Debug, PartialEq)]
pub struct PositionBucket {
	pub first: usize,
	pub last: usize,
	pub kl: f64,
	pub pearson: f64,
	pub max_relative: f64,
	pub top1: f64,
}

/// A variant run's teacher-forced logits compared with a reference run of the same text.
#[derive(Clone, Debug)]
pub struct LogitComparison {
	pub positions: Vec<PositionDelta>,
	pub kl: Spread,
	pub pearson: Spread,
	pub spearman: Spread,
	pub max_abs: Spread,
	pub max_relative: Spread,
	/// Share of positions whose reference top token is the variant's top token.
	pub top1: f64,
	/// Share of positions whose reference top token is in the variant's top five.
	pub top5: f64,
	/// Largest error of any single logit, over the reference's range at its position.
	pub relative_max: f64,
	/// 99th percentile of that error over every logit of every position.
	pub relative_p99: f64,
	pub logprob: f64,
	pub reference_logprob: f64,
	pub perplexity: f64,
	pub reference_perplexity: f64,
	/// The bound on `relative_max`, and the bound on the rise of the fitted per-position
	/// error over the run, with the fewest steps that is judged for growth.
	pub bound: f64,
	pub growth_bound: f64,
	pub growth_positions: usize,
	/// Fitted rise of `max_relative` from the first to the last position.
	pub growth: f64,
	/// Mean `max_relative` of the last quarter of positions less that of the first quarter.
	pub late_minus_early: f64,
	pub failures: Vec<String>,
}

const TOP_RANKED: usize = 100;
const ERROR_BINS: usize = 1 << 20;

fn log_softmax(logits: &[f64]) -> Vec<f64> {
	let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
	let sum = logits.iter().map(|value| (value - max).exp()).sum::<f64>();
	let shift = max + sum.ln();
	logits.iter().map(|value| value - shift).collect()
}

fn pearson(left: &[f64], right: &[f64]) -> f64 {
	let count = left.len() as f64;
	let (mean_left, mean_right) = (left.iter().sum::<f64>() / count, right.iter().sum::<f64>() / count);
	let (mut covariance, mut variance_left, mut variance_right) = (0.0, 0.0, 0.0);
	for (a, b) in left.iter().zip(right) {
		covariance += (a - mean_left) * (b - mean_right);
		variance_left += (a - mean_left) * (a - mean_left);
		variance_right += (b - mean_right) * (b - mean_right);
	}
	if variance_left == 0.0 || variance_right == 0.0 {
		// A constant vector correlates with nothing; two identical constants agree.
		return if left == right { 1.0 } else { 0.0 };
	}
	covariance / (variance_left * variance_right).sqrt()
}

/// Ranks with ties sharing their average rank.
fn ranks(values: &[f64]) -> Vec<f64> {
	let mut order = (0..values.len()).collect::<Vec<_>>();
	order.sort_by(|a, b| values[*a].total_cmp(&values[*b]));
	let mut ranks = vec![0.0; values.len()];
	let mut start = 0;
	while start < order.len() {
		let mut end = start;
		while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
			end += 1;
		}
		for index in &order[start..=end] {
			ranks[*index] = (start + end) as f64 / 2.0;
		}
		start = end + 1;
	}
	ranks
}

/// Indices of the `count` largest logits, the lowest index first among equals.
fn top_indices(logits: &[f64], count: usize) -> Vec<usize> {
	let mut order = (0..logits.len()).collect::<Vec<_>>();
	let count = count.min(order.len());
	let by = |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then(a.cmp(b));
	if count < order.len() {
		order.select_nth_unstable_by(count - 1, by);
		order.truncate(count);
	}
	order.sort_by(by);
	order
}

/// One position: the reference's logits `reference` against the variant's `variant`.
fn position_delta(reference: &[f64], variant: &[f64], target: usize, histogram: &mut ErrorHistogram) -> PositionDelta {
	let (log_reference, log_variant) = (log_softmax(reference), log_softmax(variant));
	let kl = log_reference.iter().zip(&log_variant).map(|(r, v)| r.exp() * (r - v)).sum::<f64>().max(0.0);
	let top = top_indices(reference, TOP_RANKED);
	let (ranks_reference, ranks_variant) = (ranks(&top.iter().map(|k| reference[*k]).collect::<Vec<_>>()), ranks(&top.iter().map(|k| variant[*k]).collect::<Vec<_>>()));
	let (low, high) = reference.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), value| (low.min(*value), high.max(*value)));
	let range = high - low;
	let mut max_abs = 0.0f64;
	for (a, b) in reference.iter().zip(variant) {
		let delta = (a - b).abs();
		max_abs = max_abs.max(delta);
		histogram.add(if delta == 0.0 { 0.0 } else if range > 0.0 { delta / range } else { f64::INFINITY });
	}
	let best = top_indices(reference, 1)[0];
	let variant_top = top_indices(variant, 5);
	PositionDelta {
		kl,
		pearson: pearson(reference, variant),
		spearman: pearson(&ranks_reference, &ranks_variant),
		max_abs,
		max_relative: if max_abs == 0.0 { 0.0 } else if range > 0.0 { max_abs / range } else { f64::INFINITY },
		top1: variant_top[0] == best,
		top5: variant_top.contains(&best),
		reference_logprob: log_reference[target],
		logprob: log_variant[target],
	}
}

/// Every logit's relative error, counted in fine bins that each keep the largest value
/// they saw, so a percentile is exact to the bin width and an exact zero stays zero.
struct ErrorHistogram {
	counts: Vec<u64>,
	largest: Vec<f64>,
	overflow: u64,
	overflow_largest: f64,
	total: u64,
}

impl ErrorHistogram {
	fn new() -> Self {
		Self { counts: vec![0; ERROR_BINS], largest: vec![0.0; ERROR_BINS], overflow: 0, overflow_largest: 0.0, total: 0 }
	}

	fn add(&mut self, error: f64) {
		self.total += 1;
		if error < 1.0 {
			let bin = (error * ERROR_BINS as f64) as usize;
			self.counts[bin] += 1;
			if error > self.largest[bin] {
				self.largest[bin] = error;
			}
		} else {
			self.overflow += 1;
			self.overflow_largest = self.overflow_largest.max(error);
		}
	}

	fn percentile(&self, share: f64) -> f64 {
		let wanted = ((self.total as f64 * share).ceil() as u64).max(1);
		let mut seen = 0;
		for (bin, count) in self.counts.iter().enumerate() {
			seen += count;
			if seen >= wanted {
				return self.largest[bin];
			}
		}
		self.overflow_largest
	}

	fn maximum(&self) -> f64 {
		if self.overflow > 0 { return self.overflow_largest; }
		self.counts.iter().zip(&self.largest).rev().find(|(count, _)| **count > 0).map_or(0.0, |(_, largest)| *largest)
	}
}

impl LogitComparison {
	/// Compares the variant's logits at each of `targets.len()` positions with the rows of the
	/// recorded run at `path`. `variant` fills the logits of position `p` into its argument, and
	/// `targets[p]` is the token that follows position `p` in the text. The run passes when no
	/// logit is off by more than `bound` of the reference's range at its position, and, in a run
	/// of at least `growth_positions` positions, the fitted error does not rise by more than
	/// `growth_bound` from the first position to the last.
	pub fn read(path: &str, targets: &[u32], bound: f64, growth_bound: f64, growth_positions: usize, mut variant: impl FnMut(usize, &mut Vec<f64>)) -> Result<Self, String> {
		let mut file = File::open(path).map(std::io::BufReader::new).map_err(|error| format!("cannot read reference {path}: {error}"))?;
		let mut head = [0u8; HEADER as usize];
		file.read_exact(&mut head).map_err(|_| format!("{path} is not a recipe reference file"))?;
		if &head[..8] != MAGIC {
			return Err(format!("{path} is not a recipe reference file"));
		}
		let version = read_u32(&head, 8)?;
		if version != VERSION {
			return Err(format!("{path} is reference format version {version}; this build reads {VERSION}"));
		}
		let vocabulary = read_u32(&head, 12)? as usize;
		let steps = usize::try_from(read_u64(&head, 16)?).map_err(|_| format!("{path} claims more steps than this machine can address"))?;
		if vocabulary == 0 || steps == 0 {
			return Err(format!("{path} records no steps"));
		}
		if steps != targets.len() {
			return Err(format!("{path} holds {steps} positions; this run scored {}", targets.len()));
		}
		let stride = vocabulary.checked_mul(8).and_then(|bytes| bytes.checked_add(12)).ok_or_else(|| format!("{path} names a vocabulary too large to lay out"))?;
		let expected = steps.checked_mul(stride).and_then(|bytes| bytes.checked_add(HEADER as usize)).ok_or_else(|| format!("{path} names more steps than can be laid out"))?;
		if file.get_ref().metadata().map_err(|error| error.to_string())?.len() != expected as u64 {
			return Err(format!("{path} holds {} bytes; {steps} steps of {vocabulary} logits need {expected}", file.get_ref().metadata().map_err(|error| error.to_string())?.len()));
		}
		let (mut row, mut reference, mut actual) = (vec![0u8; stride], vec![0.0f64; vocabulary], Vec::with_capacity(vocabulary));
		let (mut positions, mut histogram) = (Vec::with_capacity(steps), ErrorHistogram::new());
		for (step, target) in targets.iter().enumerate() {
			file.read_exact(&mut row).map_err(|error| format!("cannot read reference {path} step {step}: {error}"))?;
			if read_u64(&row, 0)? as usize != step {
				return Err(format!("{path} step {step} is labelled {}", read_u64(&row, 0)?));
			}
			for (k, value) in reference.iter_mut().enumerate() {
				let b = &row[12 + k * 8..20 + k * 8];
				*value = f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
				if !value.is_finite() {
					return Err(format!("{path} step {step} logit {k} is {value}"));
				}
			}
			actual.clear();
			variant(step, &mut actual);
			if actual.len() != vocabulary {
				return Err(format!("position {step} has {} logits; the reference has {vocabulary}", actual.len()));
			}
			if let Some(bad) = actual.iter().position(|value| !value.is_finite()) {
				return Err(format!("position {step} logit {bad} is {}", actual[bad]));
			}
			if *target as usize >= vocabulary {
				return Err(format!("token id {target} is outside the {vocabulary} logits"));
			}
			positions.push(position_delta(&reference, &actual, *target as usize, &mut histogram));
		}
		Ok(Self::summarize(positions, &histogram, bound, growth_bound, growth_positions))
	}

	fn summarize(positions: Vec<PositionDelta>, histogram: &ErrorHistogram, bound: f64, growth_bound: f64, growth_positions: usize) -> Self {
		let column = |value: fn(&PositionDelta) -> f64| positions.iter().map(value).collect::<Vec<_>>();
		let count = positions.len();
		let share = |wanted: fn(&PositionDelta) -> bool| positions.iter().filter(|position| wanted(position)).count() as f64 / count as f64;
		let relative = column(|position| position.max_relative);
		// Least squares line through the per-position error: its rise across the run is the growth.
		let middle = (count - 1) as f64 / 2.0;
		let mean = relative.iter().sum::<f64>() / count as f64;
		let (covariance, variance) = relative.iter().enumerate().fold((0.0, 0.0), |(covariance, variance), (index, value)| (covariance + (index as f64 - middle) * (value - mean), variance + (index as f64 - middle).powi(2)));
		let growth = if variance > 0.0 { covariance / variance * (count - 1) as f64 } else { 0.0 };
		let quarter = (count / 4).max(1);
		let late_minus_early = relative[count - quarter..].iter().sum::<f64>() / quarter as f64 - relative[..quarter].iter().sum::<f64>() / quarter as f64;
		let reference_logprob = positions.iter().map(|position| position.reference_logprob).sum::<f64>();
		let logprob = positions.iter().map(|position| position.logprob).sum::<f64>();
		let relative_max = histogram.maximum();
		let mut failures = Vec::new();
		if relative_max > bound {
			failures.push(format!("a logit is off by {relative_max:.6} of its position's reference range; the bound is {bound}"));
		}
		if count >= growth_positions && growth > growth_bound {
			failures.push(format!("the error rises by {growth:.6} of the range across {count} positions; the bound is {growth_bound}"));
		}
		Self {
			kl: Spread::of(&column(|position| position.kl)),
			pearson: Spread::of(&column(|position| position.pearson)),
			spearman: Spread::of(&column(|position| position.spearman)),
			max_abs: Spread::of(&column(|position| position.max_abs)),
			max_relative: Spread::of(&relative),
			top1: share(|position| position.top1),
			top5: share(|position| position.top5),
			relative_max,
			relative_p99: histogram.percentile(0.99),
			logprob,
			reference_logprob,
			perplexity: (-logprob / count as f64).exp(),
			reference_perplexity: (-reference_logprob / count as f64).exp(),
			bound,
			growth_bound,
			growth_positions,
			growth,
			late_minus_early,
			failures,
			positions,
		}
	}

	pub fn ok(&self) -> bool {
		self.failures.is_empty()
	}

	/// The report, one line each: every metric's spread, the agreement, the error against the bound, the
	/// change in log probability and perplexity, the growth, the positions in eight runs, and the verdict.
	pub fn lines(&self) -> Vec<String> {
		let mut lines = Vec::new();
		for (name, spread) in [("kl", self.kl), ("pearson", self.pearson), ("spearman", self.spearman), ("max-abs", self.max_abs), ("max-relative", self.max_relative)] {
			lines.push(format!("compare {name} mean {:e} median {:e} p99 {:e} min {:e} max {:e}", spread.mean, spread.median, spread.p99, spread.min, spread.max));
		}
		lines.push(format!("compare top1 {:.6} top5 {:.6}", self.top1, self.top5));
		lines.push(format!("compare relative-error max {:e} p99 {:e}", self.relative_max, self.relative_p99));
		lines.push(format!("compare logprob {:.12} reference {:.12} change {:e}", self.logprob, self.reference_logprob, self.logprob - self.reference_logprob));
		lines.push(format!("compare perplexity {:.6} reference {:.6} change {:e}", self.perplexity, self.reference_perplexity, self.perplexity - self.reference_perplexity));
		lines.push(format!("compare growth {:e} late-minus-early {:e} (judged from {} positions, bound {})", self.growth, self.late_minus_early, self.growth_positions, self.growth_bound));
		for bucket in self.buckets(8) {
			lines.push(format!("compare positions {}-{} kl {:e} pearson {:.9} max-relative {:e} top1 {:.4}", bucket.first, bucket.last, bucket.kl, bucket.pearson, bucket.max_relative, bucket.top1));
		}
		lines.extend(self.failures.iter().map(|failure| format!("compare FAIL {failure}")));
		lines.push(format!("compare bound {} {}", self.bound, if self.ok() { "PASS" } else { "FAIL" }));
		lines
	}

	/// The positions in `rows` consecutive runs, each reported by its means.
	pub fn buckets(&self, rows: usize) -> Vec<PositionBucket> {
		let count = self.positions.len();
		let rows = rows.clamp(1, count);
		(0..rows)
			.map(|row| {
				let (first, end) = (row * count / rows, (row + 1) * count / rows);
				let slice = &self.positions[first..end];
				let mean = |value: fn(&PositionDelta) -> f64| slice.iter().map(value).sum::<f64>() / slice.len() as f64;
				PositionBucket {
					first,
					last: end - 1,
					kl: mean(|position| position.kl),
					pearson: mean(|position| position.pearson),
					max_relative: slice.iter().map(|position| position.max_relative).fold(0.0, f64::max),
					top1: slice.iter().filter(|position| position.top1).count() as f64 / slice.len() as f64,
				}
			})
			.collect()
	}
}
