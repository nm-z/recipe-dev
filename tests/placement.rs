//! Layer placement through the public `recipe.place(bundle, split)` path.
//! A child process per device pair runs the model across the pair and
//! compares with the single-device `infer`; a pair whose accelerator is
//! absent is skipped.

use recipe::*;
use std::fmt::Write as _;

fn dataset() -> std::path::PathBuf {
	let path = std::env::temp_dir().join(format!("recipe-placement-{}.csv", std::process::id()));
	let mut state = 0x9e37_79b9_7f4a_7c15_u64;
	let mut random = move || {
		state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
		(state >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0
	};
	let mut text = (0..17).fold(String::new(), |mut text, column| {
		let _ = write!(text, "x{column},");
		text
	});
	text.push_str("y\n");
	for _ in 0..131 {
		let values = (0..17).map(|_| random()).collect::<Vec<_>>();
		for value in &values {
			let _ = write!(text, "{value:.6},");
		}
		let _ = writeln!(text, "{:.6}", values.iter().sum::<f64>() / 17.0);
	}
	std::fs::write(&path, text).unwrap();
	path
}

/// Runs inside the child: trains on the first selected device, then compares
/// the placed inference with the plain one, printing the evidence.
#[test]
fn placed_child() {
	if std::env::var_os("RECIPE_PLACEMENT_CHILD").is_none() {
		return;
	}
	let path = dataset();
	let data = recipe.data(path.to_string_lossy().as_ref()).target("y");
	let sample = [0.25; 17];
	for (name, model, split) in
		[("deep", recipe.model().layer(9).relu().layer(5).relu().layer(1).loss(mse), [2, 1]), ("conv-attention", recipe.model().conv(4, 3).attn(2).relu().layer(1).loss(mse), [1, 2])]
	{
		let bundle = std::env::temp_dir().join(format!("recipe-placement-{}-{name}.ogdl", std::process::id()));
		recipe.train().fp(32).lr(0.01).epochs(5).save(&bundle).run(&model, &data);
		let plain = recipe.infer(&bundle, &sample);
		let placed = recipe.place(&bundle, &split);
		let output = placed.infer(&sample);
		println!("{name} identical={} resident={:?} moved={}", output.iter().map(|v| v.to_bits()).eq(plain.iter().map(|v| v.to_bits())), placed.resident_bytes(), placed.moved_bytes());
		let _ = std::fs::remove_file(&bundle);
	}
	let bundle = std::env::temp_dir().join(format!("recipe-placement-{}-residual.ogdl", std::process::id()));
	recipe.train().fp(32).lr(0.01).epochs(1).save(&bundle).run(&recipe.model().layer(8).res([layer(8)]).layer(1).loss(mse), &data);
	let cut = std::panic::catch_unwind(|| recipe.place(&bundle, &[1, 2]));
	let message = cut.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	println!("residual error={message:?}");
	let wrong = std::panic::catch_unwind(|| recipe.place(&bundle, &[1, 1]));
	let message = wrong.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
	println!("count error={message:?}");
	let _ = std::fs::remove_file(&bundle);
	let _ = std::fs::remove_file(&path);
}

#[test]
fn placed_inference_equals_single_device() {
	let mut ran = 0;
	for pair in ["amd0,cpu", "nv0,cpu"] {
		let output = std::process::Command::new(std::env::current_exe().unwrap())
			.args(["--exact", "placed_child", "--nocapture"])
			.env("RECIPE_DEVICE", pair)
			.env("RECIPE_PLACEMENT_CHILD", "1")
			.output()
			.unwrap();
		let (stdout, stderr) = (String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
		if stderr.contains("is absent") {
			continue;
		}
		assert!(output.status.success(), "{pair} failed:\n{stdout}\n{stderr}");
		ran += 1;
		for name in ["deep", "conv-attention"] {
			assert!(stdout.contains(&format!("{name} identical=true")), "{pair}: {name} differs from the single device run\n{stdout}");
		}
		assert!(stdout.contains("deep identical=true resident=[848, 24] moved=20"), "{pair}: deep resident or moved bytes changed\n{stdout}");
		assert!(stdout.contains("residual error=\"the split cuts a second connection into block 1 from before block 1\""), "{pair}: residual cut was not refused\n{stdout}");
		assert!(stdout.contains("count error=\"the split places 2 blocks but the model has 3\""), "{pair}: block count was not checked\n{stdout}");
	}
	assert!(ran > 0, "no accelerator is present for a two-device placement");
}
