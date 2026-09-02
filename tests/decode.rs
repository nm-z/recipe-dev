//! The sampler and the decode loop through the public `recipe.sampler()` and
//! `recipe.decode(..)` path: a window model of ids trained on a fixed rule,
//! decoded greedily and compared step by step with `infer`.

use recipe::*;
use std::fmt::Write as _;

#[test]
fn sampler_orders_penalty_top_k_top_p_min_p_and_temperature() {
	let logits = [1.0, 4.0, 2.0, 4.0, -3.0];
	assert_eq!(recipe.sampler().temperature(0.0).sample(&logits, &[]), 1, "greedy takes the lowest id among ties");
	assert_eq!(recipe.sampler().temperature(2.0).top_k(1).sample(&logits, &[]), 1, "top-k of one is greedy at any temperature");
	assert_eq!(recipe.sampler().temperature(0.0).repeat(3.0, 8).sample(&logits, &[1, 1]), 3, "a repeated positive logit is divided by the penalty");
	assert_eq!(recipe.sampler().temperature(0.0).repeat(3.0, 1).sample(&logits, &[1, 3]), 1, "the window bounds the penalty");
	let mut min_p = recipe.sampler().min_p(0.9).seed(3);
	assert!((0..64).all(|_| matches!(min_p.sample(&logits, &[]), 1 | 3)), "min-p keeps only ids near the highest probability");
	let mut top_p = recipe.sampler().top_p(0.4).seed(11);
	assert!((0..64).all(|_| top_p.sample(&logits, &[]) == 1), "the highest id alone reaches the mass");
	let draw = |seed: u64| {
		let mut sampler = recipe.sampler().temperature(1.5).seed(seed);
		(0..32).map(|_| sampler.sample(&logits, &[])).collect::<Vec<_>>()
	};
	assert_eq!(draw(7), draw(7), "the same seed draws the same ids");
	assert_ne!(draw(7), draw(8), "different seeds draw different ids");
	assert!(draw(7).iter().all(|id| *id != 4), "an id 7 logits below the top never appears in 32 draws");
}

/// Windows of three ids below five whose next id is `(a + 2b + 3c) % 5`, as a
/// CSV with one-hot targets.
fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-decode-{}.csv", std::process::id()));
	let mut text = "x0,x1,x2,y0,y1,y2,y3,y4\n".to_owned();
	for a in 0..5 {
		for b in 0..5 {
			for c in 0..5 {
				let next = (a + 2 * b + 3 * c) % 5;
				let _ = writeln!(text, "{a},{b},{c},{}", (0..5).map(|id| if id == next { "1" } else { "0" }).collect::<Vec<_>>().join(","));
			}
		}
	}
	std::fs::write(&path, text).unwrap();
	path
}

#[test]
fn greedy_decode_matches_infer_step_by_step() {
	let path = dataset();
	let bundle = std::env::temp_dir().join(format!("recipe-decode-{}.ogdl", std::process::id()));
	let data = recipe.data(path.to_string_lossy().as_ref()).target(["y0", "y1", "y2", "y3", "y4"]);
	let model = recipe.model().embed(5, 4).layer(16).relu().layer(5).loss(mse);
	recipe.train().fp(32).lr(0.02).epochs(40).save(&bundle).run(&model, &data);
	let mut greedy = recipe.sampler().temperature(0.0);
	let generation = recipe.decode(&bundle, &[1, 2, 3], &mut greedy, &[], 6);
	assert_eq!(generation.ids.len(), 9, "the prompt plus the budget");
	assert_eq!(generation.step_seconds.len(), 5, "one step time per id after the prefill");
	assert!(generation.prefill_seconds > 0.0 && generation.step_seconds.iter().all(|seconds| *seconds > 0.0));
	let mut ids = vec![1, 2, 3];
	for _ in 0..6 {
		let window = ids[ids.len() - 3..].iter().map(|id| f64::from(*id)).collect::<Vec<_>>();
		let logits = recipe.infer(&bundle, &window);
		let best = (0..logits.len()).max_by(|left, right| logits[*left].total_cmp(&logits[*right]).then(right.cmp(left))).unwrap();
		ids.push(best as u32);
	}
	assert_eq!(generation.ids, ids, "decode diverged from the infer loop");
	// A stop id ends the generation at that id.
	let stop = generation.ids[3];
	let stopped = recipe.decode(&bundle, &[1, 2, 3], &mut recipe.sampler().temperature(0.0), &[stop], 6);
	assert_eq!(stopped.ids, vec![1, 2, 3, stop]);
	let _ = std::fs::remove_file(&path);
	let _ = std::fs::remove_file(&bundle);
}
