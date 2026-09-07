//! The three attention head counts in one place.
//!
//! `attn(heads)` gives the query, key and value planes the same head count.
//! `attn([q, k, v])` states the three separately, so a grouped-query block is
//! one call instead of `attn(q).kv(k)`. The two spellings must produce the same
//! model wherever they describe the same geometry, the counts must survive a
//! save, and an untied key and value count must be refused rather than silently
//! walked with one head count by the kernels.

use recipe::*;

const FEATURES: usize = 16;
const ROWS: usize = 48;

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-heads-{}", std::process::id()));
	std::fs::create_dir_all(&directory).unwrap();
	let mut text = String::new();
	for column in 0..FEATURES {
		text.push_str(&format!("x{column},"));
	}
	text.push_str("y\n");
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	for _ in 0..ROWS {
		let values = (0..FEATURES).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			text.push_str(&format!("{value},"));
		}
		let sum = values.iter().enumerate().map(|(index, value)| value * (index as f64 / 4.0).cos()).sum::<f64>();
		text.push_str(&format!("{}\n", (sum / 4.0).tanh()));
	}
	std::fs::write(directory.join("rows.csv"), text).unwrap();
	directory.to_str().unwrap().to_owned()
}

fn evidence(model: &Model) -> (u64, u64, Vec<u64>) {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let report = recipe.train().fp(32).seed(17).lr(0.01).epochs(20).stop(0.0).run(model, &data);
	(report.initial_loss().to_bits(), report.final_loss().to_bits(), report.predictions().iter().map(|value| value.to_bits()).collect())
}

fn panic_text(result: std::thread::Result<()>) -> String {
	match result.unwrap_err().downcast::<String>() {
		Ok(text) => *text,
		Err(error) => (*error.downcast::<&str>().unwrap()).to_owned(),
	}
}

/// One count and three equal counts are the same block, to the bit. This is the
/// check that the new spelling reaches the same lowering rather than a parallel
/// one that happens to train.
#[test]
fn one_count_and_three_equal_counts_agree() {
	let single = evidence(&recipe.model().attn(4).relu().layer(1).loss(mse));
	let triple = evidence(&recipe.model().attn([4, 4, 4]).relu().layer(1).loss(mse));
	assert_eq!(single, triple, "attn(4) and attn([4, 4, 4]) produced different models");
}

/// The grouped-query case: three counts must equal the `kv` spelling of the same
/// geometry, which is the one the kernels already ran before this change.
#[test]
fn three_counts_match_the_kv_spelling() {
	let chained = evidence(&recipe.model().attn(4).kv(2).relu().layer(1).loss(mse));
	let stated = evidence(&recipe.model().attn([4, 2, 2]).relu().layer(1).loss(mse));
	assert_eq!(chained, stated, "attn(4).kv(2) and attn([4, 2, 2]) produced different models");
	assert!(f64::from_bits(stated.1) < f64::from_bits(stated.0), "the grouped-query model did not train: {} to {}", f64::from_bits(stated.0), f64::from_bits(stated.1));
}

/// `kv` still sets both counts, so the shorter spelling keeps working and keeps
/// meaning what it meant. A block whose key count moved but whose value count
/// stayed behind would compile and quietly attend over the wrong planes.
#[test]
fn kv_still_sets_both_counts() {
	let stated = evidence(&recipe.model().attn([8, 2, 2]).width(4).relu().layer(1).loss(mse));
	let chained = evidence(&recipe.model().attn(8).width(4).kv(2).relu().layer(1).loss(mse));
	assert_eq!(stated, chained, "kv(2) and [8, 2, 2] disagree");
}

/// Untied key and value counts are refused at compile time, and the message
/// names all three counts so the geometry that was asked for is readable.
#[test]
fn an_untied_key_and_value_count_is_refused() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = evidence(&recipe.model().attn([8, 4, 2]).width(4).relu().layer(1).loss(mse));
	}));
	assert!(message.contains("attention key and value head counts must match"), "unexpected error: {message}");
	assert!(message.contains("8 query, 4 key and 2 value heads"), "the message did not name the three counts: {message}");
}

/// A count that does not divide the query count is refused for the value plane
/// exactly as it is for the key plane, rather than only being caught for keys.
#[test]
fn a_value_count_that_does_not_divide_is_refused() {
	let message = panic_text(std::panic::catch_unwind(|| {
		let _ = evidence(&recipe.model().attn([8, 3, 3]).width(4).relu().layer(1).loss(mse));
	}));
	assert!(message.contains("attention head partition is invalid"), "unexpected error: {message}");
	assert!(message.contains("8 query, 3 key and 3 value heads"), "the message did not name the three counts: {message}");
}

/// The three counts survive the bundle. The value count rides behind the
/// optional head width in the block record, so a model saved with a derived
/// width has to fill the width slot to reach it — and must still read back with
/// a derived width, not a width of zero.
#[test]
fn the_counts_survive_the_bundle() {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let saved = |model: &Model, tag: &str| {
		let bundle = std::env::temp_dir().join(format!("recipe-heads-{}-{tag}.ogdl", std::process::id()));
		let report = recipe.train().fp(32).seed(17).lr(0.01).epochs(20).stop(0.0).save(&bundle).run(model, &data);
		let output = recipe.infer(&bundle, &[0.25; FEATURES]);
		let _ = std::fs::remove_file(&bundle);
		(report.initial_loss().to_bits(), report.final_loss().to_bits(), output[0].to_bits())
	};
	// A derived head width, so the width slot is empty and the value count is the
	// only thing behind it. Reading the value count as the width, or losing it,
	// gives a different geometry and a different inference.
	let stated = saved(&recipe.model().attn([4, 2, 2]).relu().layer(1).loss(mse), "stated");
	let chained = saved(&recipe.model().attn(4).kv(2).relu().layer(1).loss(mse), "chained");
	assert_eq!(stated, chained, "the three-count record did not reload as the same geometry as kv(2)");
	assert!(f64::from_bits(stated.1) < f64::from_bits(stated.0), "the saved model did not train");

	// The other branch of the width slot: a declared width with the value count
	// behind it.
	let wide_stated = saved(&recipe.model().attn([8, 2, 2]).width(4).relu().layer(1).loss(mse), "wide-stated");
	let wide_chained = saved(&recipe.model().attn(8).width(4).kv(2).relu().layer(1).loss(mse), "wide-chained");
	assert_eq!(wide_stated, wide_chained, "a declared width and a value count did not reload together");
}
