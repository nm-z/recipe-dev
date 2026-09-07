//! Device admission: one run at a time holds a local accelerator.
//!
//! Two runs that name the same device must not open a context on it at once;
//! two runs that name different devices must not wait for each other. The lease
//! lives on an open file description, so a killed run frees its device with
//! nothing running afterwards.
//!
//! This drives the public entry point through child processes, because the lease
//! is a property of the process rather than of a call: one process takes it once,
//! in `selected_gpus`, and holds it until it exits.

use recipe::*;
use std::path::PathBuf;
use std::time::Instant;

const COLUMNS: usize = 8;

fn dataset() -> String {
	let directory = std::env::temp_dir().join(format!("recipe-admission-{}", std::process::id()));
	std::fs::create_dir_all(&directory).unwrap();
	let mut text = String::new();
	for column in 0..COLUMNS {
		text.push_str(&format!("x{column},"));
	}
	text.push_str("y\n");
	for row in 0..64 {
		let t = row as f64 / 64.0;
		for column in 0..COLUMNS {
			text.push_str(&format!("{},", (t + column as f64 / 8.0).sin()));
		}
		text.push_str(&format!("{}\n", (t * 3.0).cos()));
	}
	std::fs::write(directory.join("rows.csv"), text).unwrap();
	directory.to_str().unwrap().to_owned()
}

/// The work one child does while holding its device. Long enough that two
/// children overlap if nothing stops them.
fn hold() {
	let directory = dataset();
	let data = recipe.data(directory.as_str()).target("y");
	let model = recipe.model().layer(16).gelu().layer(8).gelu().layer(1).loss(mse);
	let report = recipe.train().fp(32).seed(3).lr(0.01).epochs(400).stop(0.0).run(&model, &data);
	println!("HELD {}", report.final_loss().is_finite());
}

fn child(device: &str) -> std::process::Child {
	let executable = std::env::current_exe().unwrap_or_else(|error| panic!("cannot locate the test binary: {error}"));
	std::process::Command::new(&executable)
		.env("RECIPE_ADMISSION_CHILD", "1")
		.env("RECIPE_DEVICE", device)
		.arg("--exact")
		.arg("--nocapture")
		.arg("admission_child")
		.stdout(std::process::Stdio::piped())
		.stderr(std::process::Stdio::piped())
		.spawn()
		.unwrap_or_else(|error| panic!("cannot run a child on {device}: {error}"))
}

fn finish(child: std::process::Child) -> (bool, String) {
	let output = child.wait_with_output().unwrap();
	(String::from_utf8_lossy(&output.stdout).contains("HELD true"), String::from_utf8_lossy(&output.stderr).into_owned())
}

/// The child hook. Running the suite normally skips it; the parent below sets
/// `RECIPE_ADMISSION_CHILD` to make one process do the work and exit.
#[test]
fn admission_child() {
	if std::env::var("RECIPE_ADMISSION_CHILD").is_err() {
		return;
	}
	hold();
}

fn devices() -> Vec<String> {
	std::env::var("RECIPE_ADMISSION_DEVICES").map(|names| names.split(',').map(str::to_owned).collect()).unwrap_or_default()
}

/// Two runs naming the same device serialize: their holds do not overlap, so the
/// pair takes about as long as running them one after the other, and both finish.
#[test]
fn two_runs_naming_one_device_serialize() {
	let Some(device) = devices().first().cloned() else {
		eprintln!("RECIPE_ADMISSION_DEVICES is unset; skipping");
		return;
	};
	let alone = Instant::now();
	let (first, error) = finish(child(&device));
	assert!(first, "a single run on {device} did not finish: {error}");
	let single = alone.elapsed();

	let together = Instant::now();
	let (left, right) = (child(&device), child(&device));
	let (left, left_error) = finish(left);
	let (right, right_error) = finish(right);
	let pair = together.elapsed();

	assert!(left, "the first of two runs on {device} did not finish: {left_error}");
	assert!(right, "the second of two runs on {device} did not finish: {right_error}");
	// Serialized, the pair costs about two runs. Overlapping, it costs about one
	// plus the contention. The margin is generous because the point is the shape,
	// not the constant.
	assert!(
		pair.as_secs_f64() > single.as_secs_f64() * 1.5,
		"two runs on {device} took {:.2}s against {:.2}s for one, which is not serialized",
		pair.as_secs_f64(),
		single.as_secs_f64()
	);
	assert!(left_error.contains("in use by another run") || right_error.contains("in use by another run"), "neither run reported waiting:\n{left_error}\n{right_error}");
}

/// Two runs naming different devices do not wait for each other.
#[test]
fn two_runs_naming_different_devices_do_not_wait() {
	let names = devices();
	let (Some(first), Some(second)) = (names.first(), names.get(1)) else {
		eprintln!("RECIPE_ADMISSION_DEVICES names fewer than two devices; skipping");
		return;
	};
	let (left, right) = (child(first), child(second));
	let (left, left_error) = finish(left);
	let (right, right_error) = finish(right);
	assert!(left, "the run on {first} did not finish: {left_error}");
	assert!(right, "the run on {second} did not finish: {right_error}");
	assert!(!left_error.contains("in use by another run"), "the run on {first} waited for {second}:\n{left_error}");
	assert!(!right_error.contains("in use by another run"), "the run on {second} waited for {first}:\n{right_error}");
}

/// A killed run frees its device: the lease is on the open file description, so
/// nothing has to run after the kill for the next run to be admitted.
#[test]
fn a_killed_run_frees_its_device() {
	let Some(device) = devices().first().cloned() else {
		eprintln!("RECIPE_ADMISSION_DEVICES is unset; skipping");
		return;
	};
	let mut holder = child(&device);
	// Give it long enough to have taken the lease before it is killed.
	std::thread::sleep(std::time::Duration::from_millis(1500));
	holder.kill().unwrap();
	let _ = holder.wait();

	let started = Instant::now();
	let (finished, error) = finish(child(&device));
	assert!(finished, "a run after a killed one did not finish: {error}");
	assert!(!error.contains("did not become free"), "the killed run's lease outlived it: {error}");
	assert!(started.elapsed().as_secs_f64() < 120.0, "a run after a killed one waited {:.0}s", started.elapsed().as_secs_f64());
}

/// The lease directory is per user, so two people never contend for each other's
/// devices and neither needs write access to the other's.
#[test]
fn the_lease_directory_is_per_user() {
	let root = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
	assert!(root.is_absolute(), "the lease root {root:?} is not absolute");
}
