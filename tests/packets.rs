//! Reported packets: the exact public program of each performance issue must
//! finish training, saving, and inference within the slow-runtime threshold
//! that filed it.
//!
//! Each packet runs in a child process of this binary on its reported device,
//! so a stall or abort in one packet cannot hide the others. The parent kills a
//! child that outlives the threshold and reports the packet as a failure.

use recipe::*;
use std::fmt::Write as _;
use std::io::Write as _;

// The libtest harness captures the print macros on passing tests, so write reports straight to the inherited stderr descriptor.
fn report(text: String) {
	use std::os::fd::FromRawFd;
	let mut stderr = unsafe { std::fs::File::from_raw_fd(2) };
	let _ = stderr.write_all(text.as_bytes());
	std::mem::forget(stderr);
}

/// The elapsed wall time the composition traversal allows one public run.
const THRESHOLD: std::time::Duration = std::time::Duration::from_secs(60);

struct Packet {
	issue: u32,
	/// The reported device: `cpu`, or a local GPU name such as `nv0`. A packet
	/// whose device is absent from this host is reported as skipped.
	device: &'static str,
	run: fn(&str) -> TrainingReport,
}

const PACKETS: &[Packet] = &[Packet { issue: 511, device: "cpu", run: packet_511 }, Packet { issue: 512, device: "cpu", run: packet_512 }, Packet { issue: 519, device: "cpu", run: packet_519 }];

/// Quantized residual convolution block feeding a random forest on the z-scored
/// temporal chronological splits.
fn packet_512(bundle: &str) -> TrainingReport {
	let data = recipe.data("data/temporal/chronological_splits").target("target").norm(z_score);
	let model = recipe.model().layer(8).res([conv(8, 1), conv(8, 1)]).silu().qi(4).k.s.forest(2).silu().loss(rmse);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(8243371770253).epochs(1).log(all).tf(32).save(bundle).run(&model, &data);
	let output = recipe.infer(bundle, &[0.0; 20]);
	assert!(output.iter().all(|value| value.is_finite()), "inference is not finite: {output:?}");
	report
}

/// Quantized residual layer and convolution blocks with batch normalization on
/// the z-scored temporal chronological splits.
fn packet_511(bundle: &str) -> TrainingReport {
	let data = recipe.data("data/temporal/chronological_splits").target("target").norm(z_score);
	let model = recipe.model().layer(8).res([layer(8), layer(8)]).cos().norm(batch).qi(5).0.layer(8).res([conv(8, 1), conv(8, 1)]).norm(batch).iq(2).xxs.loss(huber);
	let report = recipe.train().optimizer(adamw).lr(0.0001).seed(39864877594399).epochs(1).log(all).f(6, 9).save(bundle).run(&model, &data);
	let output = recipe.infer(bundle, &[0.0; 20]);
	assert!(output.iter().all(|value| value.is_finite()), "inference is not finite: {output:?}");
	report
}

/// Quantized layer and pool model on the temporal chronological splits.
fn packet_519(bundle: &str) -> TrainingReport {
	let data = recipe.data("data/temporal/chronological_splits").target("target");
	let model = recipe.model().layer(8).qi(8).1.pool(2).selu().qi(3).k.loss(huber);
	let report = recipe.train().optimizer(adamw).lr(0.001).seed(8431471041318).epochs(1).log(all).tf(32).save(bundle).run(&model, &data);
	let output = recipe.infer(bundle, &[0.0; 20]);
	assert!(output.iter().all(|value| value.is_finite()), "inference is not finite: {output:?}");
	report
}

/// Runs one packet on its reported device and returns its wall time, or `None`
/// when the device is absent from this host.
fn child(packet: &Packet) -> std::result::Result<Option<f64>, String> {
	let executable = std::env::current_exe().map_err(|error| format!("cannot locate the test binary: {error}"))?;
	let mut command = std::process::Command::new(executable);
	command
		.env("RECIPE_PACKET", packet.issue.to_string())
		.env("RECIPE_DEVICE", packet.device)
		.args(["--exact", "--nocapture", "reported_packets_finish_under_threshold"])
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped());
	if packet.device == "cpu" {
		command.env("RECIPE_FORCE_CPU", "1");
	}
	let start = std::time::Instant::now();
	let mut child = command.spawn().map_err(|error| format!("cannot start the packet: {error}"))?;
	while child.try_wait().map_err(|error| format!("cannot wait for the packet: {error}"))?.is_none() {
		if start.elapsed() > THRESHOLD {
			let _ = child.kill();
			let _ = child.wait();
			return Err(format!("still running after {} s", THRESHOLD.as_secs()));
		}
		std::thread::sleep(std::time::Duration::from_millis(50));
	}
	let seconds = start.elapsed().as_secs_f64();
	let output = child.wait_with_output().map_err(|error| format!("cannot read the packet: {error}"))?;
	let text = String::from_utf8_lossy(&output.stdout);
	let errors = String::from_utf8_lossy(&output.stderr);
	let epoch = text.lines().find_map(|line| line.strip_prefix("PACKET ")).and_then(|line| line.parse::<f64>().ok());
	match (output.status.success(), epoch) {
		(true, Some(epoch)) => {
			report(format!("  #{:<4} {:<5} {seconds:>8.3} s wall {epoch:>10.6} s epoch\n", packet.issue, packet.device));
			Ok(Some(seconds))
		}
		_ if errors.contains(&format!("GPU {:?} is absent", packet.device)) => {
			report(format!("  #{:<4} {:<5} skipped, device absent\n", packet.issue, packet.device));
			Ok(None)
		}
		_ => Err(format!("exited with {} after {seconds:.3} s:\n{text}\n{errors}", output.status)),
	}
}

#[test]
fn reported_packets_finish_under_threshold() {
	if let Ok(issue) = std::env::var("RECIPE_PACKET") {
		let packet = PACKETS.iter().find(|packet| packet.issue.to_string() == issue).unwrap_or_else(|| panic!("unknown packet {issue:?}"));
		let bundle = std::env::temp_dir().join(format!("recipe-packet-{issue}-{}.ogdl", std::process::id()));
		let report = (packet.run)(bundle.to_str().unwrap_or_else(|| panic!("bundle path is not UTF-8")));
		let _ = std::fs::remove_file(&bundle);
		assert!(report.final_loss().is_finite(), "final loss is not finite");
		println!("PACKET {}", report.epoch_seconds());
		return;
	}
	report(String::from("\nreported packets\n"));
	let mut failures = Vec::new();
	for packet in PACKETS {
		if let Err(failure) = child(packet) {
			let mut message = String::new();
			let _ = write!(message, "#{}: {failure}", packet.issue);
			failures.push(message);
		}
	}
	assert!(failures.is_empty(), "{} of {} packets crossed the threshold:\n{}", failures.len(), PACKETS.len(), failures.join("\n"));
}
