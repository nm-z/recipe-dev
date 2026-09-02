//! Speculative decode through the public `recipe.speculate(..)` path: a
//! draft bundle proposes a lookahead of ids, the main bundle verifies them
//! in one batched forward, and the ids equal a plain greedy decode.

use recipe::*;
use std::fmt::Write as _;

/// Windows of three ids below five whose next id is `(a + 2b + 3c) % 5`.
fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-speculate-{}.csv", std::process::id()));
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
fn speculative_ids_equal_greedy_decode_for_any_draft() {
	let path = dataset();
	let data = recipe.data(path.to_string_lossy().as_ref()).target(["y0", "y1", "y2", "y3", "y4"]);
	let main = std::env::temp_dir().join(format!("recipe-speculate-main-{}.ogdl", std::process::id()));
	recipe.train().fp(32).lr(0.02).epochs(120).save(&main).run(&recipe.model().embed(5, 4).layer(16).relu().layer(5).loss(mse), &data);
	// A draft that saw one epoch proposes mostly wrong ids; the main model is
	// its own perfect draft.
	let weak = std::env::temp_dir().join(format!("recipe-speculate-weak-{}.ogdl", std::process::id()));
	recipe.train().fp(32).lr(0.02).epochs(1).save(&weak).run(&recipe.model().embed(5, 2).layer(5).loss(mse), &data);
	let plain = recipe.decode(&main, &[1, 2, 3], &mut recipe.sampler().temperature(0.0), &[], 24);
	let with_weak = recipe.speculate(&main, &weak, &[1, 2, 3], 3, &[], 24);
	let with_self = recipe.speculate(&main, &main, &[1, 2, 3], 3, &[], 24);
	assert_eq!(with_weak.ids, plain.ids, "a rejected draft changed the ids");
	assert_eq!(with_self.ids, plain.ids, "an accepted draft changed the ids");
	assert_eq!(with_weak.ids.len(), 27, "the prompt plus the budget");
	assert_eq!((with_self.accepted, with_self.rejected), (18, 0), "a perfect draft is accepted at every step");
	assert_eq!(with_self.forwards, 6, "every batch of four ids is one main forward");
	assert!(with_weak.forwards > with_self.forwards, "rejections cost main forwards");
	assert_eq!(with_weak.accepted + with_weak.rejected + with_weak.forwards, with_weak.ids.len() - 3 + with_weak.rejected, "every id is a proposal or a main choice");
	assert!(with_self.seconds > 0.0 && with_weak.seconds > 0.0);
	// A stop id ends the ids at that id even inside an accepted batch.
	let stop = plain.ids[5];
	let stopped = recipe.speculate(&main, &main, &[1, 2, 3], 3, &[stop], 24);
	let first = 3 + plain.ids[3..].iter().position(|id| *id == stop).unwrap();
	assert_eq!(stopped.ids, plain.ids[..=first].to_vec());
	for file in [path, main, weak] {
		let _ = std::fs::remove_file(file);
	}
}
