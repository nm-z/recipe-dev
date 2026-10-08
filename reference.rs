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
