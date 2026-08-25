//! Public end-to-end acceptance for tri-state `multi-device` placement.
//!
//! The one test exercises the exact `recipe file.rs` entrypoint on a real
//! autoregressive text workload at fp8 and asserts the behavior of whichever
//! `multi-device` policy the current build carries, so the complete acceptance
//! is this test under each of the three configurations. While the Recipe
//! process runs, the worker host's device is sampled independently over ssh,
//! and the whole run from Recipe process start to the final verdict must
//! finish within twenty seconds.
use std::io::Read as _;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const BUDGET_SECONDS: f64 = 20.0;
const EPOCHS: usize = 6;

fn manifest(key: &str) -> String {
	let path = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
	let manifest = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("cannot read {path}: {error}"));
	manifest.lines().find_map(|line| line.trim().strip_prefix(&format!("{key} = "))).unwrap_or_else(|| panic!("Cargo.toml does not configure {key}")).trim_matches('"').to_owned()
}

fn hostname() -> String {
	let host = std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap_or_else(|error| panic!("cannot read hostname: {error}"));
	host.trim().to_owned()
}

fn fixture() -> std::path::PathBuf {
	let corpus = concat!(env!("CARGO_MANIFEST_DIR"), "/data/text/autoregressive_lines.txt");
	let text = std::fs::read_to_string(corpus).unwrap_or_else(|error| panic!("cannot read {corpus}: {error}"));
	let slice = text.lines().take(24).map(|line| format!("{line}\n")).collect::<String>();
	assert!(slice.len() > 512, "text fixture is unexpectedly small");
	let path = std::env::temp_dir().join(format!("recipe-73-text-{}.txt", std::process::id()));
	std::fs::write(&path, slice).unwrap_or_else(|error| panic!("cannot write fixture: {error}"));
	path
}

fn workload(fixture: &std::path::Path) -> std::path::PathBuf {
	let source = format!("use recipe::*;\nfn main() {{\n\tlet data = recipe.data(auto).set({fixture:?});\n\tlet model = recipe.model().lstm(8).layer(1).loss(mse);\n\tlet report = recipe.train().seed(17).epochs({EPOCHS}).fp(8).run(&model, &data);\n\tprintln!(\"report loss={{:.12}} r2={{:.12}} epoch-seconds={{:.6}}\", report.final_loss(), report.r2(), report.epoch_seconds());\n}}\n");
	let path = std::env::temp_dir().join(format!("recipe-73-workload-{}.rs", std::process::id()));
	std::fs::write(&path, source).unwrap_or_else(|error| panic!("cannot write workload: {error}"));
	path
}

struct WorkerSample {
	memory: u64,
	workers: usize,
	compute_apps: usize,
}

fn sample_worker(host: &str) -> WorkerSample {
	let output = Command::new("ssh").args(["-o", "BatchMode=yes", host, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; pgrep -fc '[r]ecipe --worker' || true; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"]).output().unwrap_or_else(|error| panic!("cannot sample worker host {host}: {error}"));
	assert!(output.status.success(), "sampling worker host {host} failed: {}", String::from_utf8_lossy(&output.stderr));
	let text = String::from_utf8_lossy(&output.stdout);
	let mut lines = text.lines();
	let mut field = |role: &str| lines.next().unwrap_or_else(|| panic!("worker sample lacks {role}")).trim().parse::<u64>().unwrap_or_else(|error| panic!("worker {role} sample is malformed: {error}"));
	WorkerSample { memory: field("memory"), workers: field("workers") as usize, compute_apps: field("compute apps") as usize }
}

/// One route line: the named candidate and its four predicted components.
struct Candidate {
	names: String,
	total: f64,
	components: [f64; 4],
}

fn parse_routes(stderr: &str) -> (Vec<Candidate>, Option<String>) {
	let mut candidates = Vec::new();
	let mut selected = None;
	for line in stderr.lines() {
		if let Some(rest) = line.strip_prefix("route ") {
			let fields = rest.split_whitespace().collect::<Vec<_>>();
			assert!(fields.len() == 16, "route line has an unexpected shape: {line}");
			let seconds = |field: &str| field.trim_end_matches('s').parse::<f64>().unwrap_or_else(|error| panic!("route seconds are malformed in {line}: {error}"));
			candidates.push(Candidate { names: fields[0].to_owned(), total: seconds(fields[3]), components: [seconds(fields[6]), seconds(fields[9]), seconds(fields[12]), seconds(fields[15])] });
		}
		if let Some(rest) = line.strip_prefix("selected route ") {
			selected = rest.split_whitespace().next().map(str::to_owned);
		}
	}
	(candidates, selected)
}

#[test]
fn placement_policy_selects_and_executes_the_route() {
	let started = Instant::now();
	let policy = manifest("multi-device");
	let worker = manifest("workers");
	let worker = worker.split([' ', ';', ',']).find(|value| !value.is_empty()).unwrap_or_else(|| panic!("workers does not name a host"));
	let local = hostname();
	let fixture = fixture();
	let workload = workload(&fixture);
	let idle = sample_worker(worker);
	assert!(idle.workers == 0, "worker host {worker} already runs a Recipe worker");

	let mut child = Command::new(env!("CARGO_BIN_EXE_recipe")).arg(&workload).stdout(Stdio::piped()).stderr(Stdio::piped()).spawn().unwrap_or_else(|error| panic!("cannot start the public Recipe entrypoint: {error}"));
	let running = AtomicBool::new(true);
	let (samples, output) = std::thread::scope(|scope| {
		let sampler = scope.spawn(|| {
			let mut samples = Vec::new();
			while running.load(Ordering::Acquire) {
				samples.push(sample_worker(worker));
			}
			samples
		});
		let mut stdout = String::new();
		let mut stderr = String::new();
		child.stdout.take().unwrap_or_else(|| panic!("child stdout is absent")).read_to_string(&mut stdout).unwrap_or_else(|error| panic!("cannot read child stdout: {error}"));
		child.stderr.take().unwrap_or_else(|| panic!("child stderr is absent")).read_to_string(&mut stderr).unwrap_or_else(|error| panic!("cannot read child stderr: {error}"));
		let status = child.wait().unwrap_or_else(|error| panic!("cannot wait for the Recipe run: {error}"));
		running.store(false, Ordering::Release);
		(sampler.join().unwrap_or_else(|_| panic!("worker sampler panicked")), (status, stdout, stderr))
	});
	let (status, stdout, stderr) = output;
	let elapsed = started.elapsed().as_secs_f64();
	std::fs::remove_file(&fixture).ok();
	std::fs::remove_file(&workload).ok();
	assert!(status.success(), "the Recipe run failed under multi-device = {policy}:\n{stderr}");

	let report = stdout.lines().find(|line| line.starts_with("report loss=")).unwrap_or_else(|| panic!("the run printed no report:\n{stdout}"));
	let loss = report.split('=').nth(1).and_then(|field| field.split_whitespace().next()).and_then(|value| value.parse::<f64>().ok()).unwrap_or_else(|| panic!("report loss is malformed: {report}"));
	assert!(loss.is_finite(), "training produced a nonfinite loss");
	let device_line = stderr.lines().find(|line| line.ends_with(".fp8")).unwrap_or_else(|| panic!("the run printed no device line:\n{stderr}"));
	let local_prefix = format!("{local}:");
	let worker_prefix = format!("{worker}:");
	let peak_memory = samples.iter().map(|sample| sample.memory).max().unwrap_or(idle.memory);
	let peak_workers = samples.iter().map(|sample| sample.workers).max().unwrap_or(0);
	let peak_apps = samples.iter().map(|sample| sample.compute_apps).max().unwrap_or(0);
	println!("policy={policy} elapsed={elapsed:.3}s device-line={device_line} samples={} idle-memory={} peak-memory={peak_memory} peak-workers={peak_workers} peak-compute-apps={peak_apps}", samples.len(), idle.memory);

	let (candidates, selected) = parse_routes(&stderr);
	match policy.as_str() {
		"false" => {
			assert!(device_line.starts_with(&local_prefix) && !device_line.contains(','), "multi-device = false must train on the local device alone, not {device_line}");
			assert!(!stderr.contains(&worker_prefix), "multi-device = false must not name the worker device:\n{stderr}");
			assert!(peak_workers == 0, "multi-device = false must not start a worker on {worker}");
			assert!(peak_apps == idle.compute_apps, "multi-device = false must not execute on {worker}: idle {} compute apps, saw {peak_apps}", idle.compute_apps);
			assert!(peak_memory <= idle.memory + 16, "multi-device = false must not allocate on {worker}: idle {} MiB, saw {peak_memory} MiB", idle.memory);
		}
		"true" => {
			assert!(device_line.contains(&local_prefix) && device_line.contains(&worker_prefix) && device_line.contains('#'), "multi-device = true must shard across the local and the worker device, not {device_line}");
			assert!(peak_workers >= 1, "multi-device = true must run a worker on {worker}");
			assert!(peak_apps > idle.compute_apps, "multi-device = true must execute on {worker}'s device: idle {} compute apps, saw {peak_apps}", idle.compute_apps);
			assert!(peak_memory > idle.memory + 16, "multi-device = true must allocate on {worker}'s device: idle {} MiB, saw {peak_memory} MiB", idle.memory);
		}
		"auto" => {
			assert!(candidates.len() >= 3, "auto must price every candidate route, saw {}:\n{stderr}", candidates.len());
			for candidate in &candidates {
				let sum = candidate.components.iter().sum::<f64>();
				assert!((sum - candidate.total).abs() <= 1e-9 + candidate.total * 1e-6, "route {} components {sum} disagree with its total {}", candidate.names, candidate.total);
			}
			let fastest = candidates.iter().min_by(|left, right| left.total.total_cmp(&right.total)).unwrap_or_else(|| panic!("no candidate routes"));
			let selected = selected.unwrap_or_else(|| panic!("auto printed no selected route:\n{stderr}"));
			assert!(selected == fastest.names, "auto selected {selected}, but the lowest predicted complete epoch is {} at {:.9}s", fastest.names, fastest.total);
			let executed = device_line.trim_end_matches(".fp8").split(',').map(|device| device.split('#').next().unwrap_or(device)).collect::<Vec<_>>().join(",");
			assert!(executed == selected, "auto executed {executed}, not its selected route {selected}");
			assert!(selected.starts_with(&local_prefix) && !selected.contains(','), "this workload's complete epoch is fastest on the local device alone, but auto selected {selected}");
		}
		other => panic!("multi-device = {other} is not a valid policy"),
	}
	assert!(elapsed <= BUDGET_SECONDS, "acceptance took {elapsed:.3}s, over the {BUDGET_SECONDS}s budget");
}
