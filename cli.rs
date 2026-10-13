use std::{fs, path::Path, path::PathBuf, process::Command};

const USAGE: &str = "usage: recipe [run] <source.rs> [--device <[node:]device[.device...]>] [--cfg <precision table>] [--ctx <positions>] [--ttl <minutes>] [-p <text>] [export]\n\trecipe stats|keys <file.gguf>\n\trecipe schema probe <dataset>\n\trecipe schema finalize <proposal.json> <answers.json> <schema.json>";

fn invalid(message: &str) -> ! {
	eprintln!("{message}");
	std::process::exit(2)
}

fn mapped(mapping: Option<&'static str>, suffix: &str) -> Vec<(String, &'static str)> {
	mapping.into_iter().flat_map(|values| values.split(';')).filter_map(|value| value.split_once('=')).map(|(target, path)| (format!("{target}.{suffix}"), path)).collect()
}

fn export(source: &Path, selected: Option<&str>) {
	fs::metadata(source).unwrap_or_else(|error| panic!("cannot inspect {}: {error}", source.display()));
	let device = selected.map(|name| name.rsplit(':').next().unwrap_or(name));
	if device.is_some_and(|name| name != "cpu" && !name.starts_with("amd") && !name.starts_with("nv")) {
		invalid("export device must be cpu, an amd device, or an nv device")
	}
	let mut artifacts = mapped(option_env!("RECIPE_HSA_CODE_OBJECTS"), "hsaco");
	artifacts.push(("cpu.a".to_owned(), concat!(env!("OUT_DIR"), "/librecipe_cpu.a")));
	artifacts.extend(mapped(option_env!("RECIPE_HSA_ASSEMBLIES"), "amd.s"));
	artifacts.extend(option_env!("RECIPE_NV_PTX").map(|path| ("ptx".to_owned(), path)));
	artifacts.retain(|(extension, _)| {
		device.is_none_or(|name| match name {
			"cpu" => extension == "cpu.a",
			name if name.starts_with("amd") => extension.ends_with(".hsaco") || extension.ends_with(".amd.s"),
			_ => extension == "ptx",
		})
	});
	assert!(!artifacts.is_empty(), "Recipe artifacts for {} were not compiled", selected.unwrap_or("this build"));
	for (extension, compiled) in artifacts {
		let output = source.with_file_name(format!("recipe.{extension}"));
		fs::copy(compiled, &output).unwrap_or_else(|error| panic!("cannot export {}: {error}", output.display()));
		eprintln!("exported: {}", output.display());
	}
}

fn library_path(directory: &Path) -> PathBuf {
	let direct = directory.join("librecipe.rlib");
	let dependencies = directory.join("deps");
	let mut selected = direct.clone();
	let mut selected_time = direct.metadata().and_then(|metadata| metadata.modified()).ok();
	if let Ok(entries) = fs::read_dir(&dependencies) {
		for candidate in entries.flatten() {
			let path = candidate.path();
			let name = path.file_name().map(|value| value.to_string_lossy()).unwrap_or_default();
			if !name.starts_with("librecipe-") || !name.ends_with(".rlib") {
				continue;
			}
			let Ok(modified) = path.metadata().and_then(|metadata| metadata.modified()) else { continue };
			if selected_time.is_none_or(|time| modified > time) {
				selected_time = Some(modified);
				selected = path;
			}
		}
	}
	selected
}

fn child_status(command: &mut Command) -> std::io::Result<std::process::ExitStatus> {
	#[cfg(target_os = "linux")]
	{
		use std::os::unix::process::CommandExt;
		const PR_SET_PDEATHSIG: i32 = 1;
		const SIGKILL: std::ffi::c_ulong = 9;
		unsafe extern "C" {
			fn prctl(option: i32, ...) -> i32;
			fn getppid() -> i32;
			fn _exit(status: i32) -> !;
		}
		let parent = std::process::id() as i32;
		unsafe {
			command.pre_exec(move || {
				// Direct children must stop when their CLI exits, even if they ignore SIGTERM.
				if prctl(PR_SET_PDEATHSIG, SIGKILL, 0 as std::ffi::c_ulong, 0 as std::ffi::c_ulong, 0 as std::ffi::c_ulong) == -1 {
					return Err(std::io::Error::last_os_error());
				}
				// The parent can exit between fork and binding the death signal. Do not exec in that case.
				if getppid() != parent {
					_exit(1);
				}
				Ok(())
			});
		}
	}
	command.status()
}

fn script_manifest(source: &Path) -> Option<String> {
	let text = fs::read_to_string(source).unwrap_or_else(|error| panic!("cannot read {}: {error}", source.display()));
	let (mut manifest, mut opened, mut closed) = (String::new(), false, false);
	for (index, line) in text.lines().enumerate() {
		let line = line.trim_start();
		if line.is_empty() || (index == 0 && line.starts_with("#!") && !line.starts_with("#![")) { continue; }
		let Some(comment) = line.strip_prefix("//!") else { break };
		let comment = comment.strip_prefix(' ').unwrap_or(comment);
		if comment.trim() == "```cargo" {
			if opened { invalid("a Recipe script may declare only one cargo manifest"); }
			opened = true;
		} else if opened && !closed {
			if comment.trim() == "```" { closed = true; }
			else { manifest.push_str(comment); manifest.push('\n'); }
		}
	}
	if opened && !closed { invalid("the script's cargo manifest has no closing ``` line"); }
	opened.then_some(manifest)
}

fn compile_script(source: &Path, library: &Path, dependencies: &Path, output: &Path) -> std::process::ExitStatus {
	let Some(manifest) = script_manifest(source) else {
		return child_status(Command::new("rustc").arg("--edition=2024").arg(source).arg("--extern").arg(format!("recipe={}", library.display()))
			.arg("-L").arg(format!("dependency={}", dependencies.display())).arg("-o").arg(output)).expect("cannot execute rustc");
	};
	use std::hash::{Hash, Hasher};
	let source = fs::canonicalize(source).expect("cannot resolve Recipe script");
	let mut key = std::hash::DefaultHasher::new();
	source.hash(&mut key);
	let root = std::env::var_os("CARGO_HOME").map(PathBuf::from)
		.or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
		.unwrap_or_else(std::env::temp_dir).join("recipe-scripts").join(format!("{:016x}", key.finish()));
	fs::create_dir_all(&root).expect("cannot create Recipe script cache");
	let lock = fs::OpenOptions::new().create(true).truncate(false).write(true).open(root.join("compile.lock")).expect("cannot open Recipe script cache lock");
	lock.lock().expect("cannot lock Recipe script cache");
	let path = root.join("Cargo.toml");
	// Keep the original source path so modules, include! and compiler diagnostics
	// retain their script-relative paths. Cargo owns dependency resolution and its lockfile.
	let text = format!("[package]\nname = \"recipe-script\"\nversion = \"0.0.0\"\nedition = \"2024\"\nbuild = false\n\n[workspace]\n\n[[bin]]\nname = \"recipe-script\"\npath = {:?}\n\n{manifest}", source.to_string_lossy());
	if fs::read_to_string(&path).ok().as_deref() != Some(&text) { fs::write(&path, text).expect("cannot write Recipe script manifest"); }
	// The prebuilt library is not a Cargo dependency. Its identity must still
	// invalidate the final script when the installed Recipe build changes.
	library.hash(&mut key);
	let metadata = fs::metadata(library).expect("cannot inspect Recipe library");
	metadata.len().hash(&mut key);
	metadata.modified().expect("cannot inspect Recipe library modification time").hash(&mut key);
	let target = root.join("target");
	let status = child_status(Command::new("cargo").arg("rustc").arg("--release").arg("--manifest-path").arg(&path).arg("--target-dir").arg(&target)
		.arg("--bin").arg("recipe-script").arg("--").arg("--extern").arg(format!("recipe={}", library.display())).arg("-L").arg(format!("dependency={}", dependencies.display()))
		.arg("--cfg").arg(format!("recipe_library=\"{:016x}\"", key.finish()))).expect("cannot execute cargo for Recipe script dependencies");
	if status.success() {
		fs::copy(target.join("release").join(format!("recipe-script{}", std::env::consts::EXE_SUFFIX)), output).expect("cannot copy compiled Recipe script");
	}
	status
}

fn run(source: &Path, device: Option<&str>, config: Option<&str>, settings: &[(String, String)], arguments: &[String]) {
	let directory = std::env::current_exe().expect("cannot locate recipe").parent().expect("recipe has no parent directory").to_owned();
	let library = library_path(&directory);
	let dependencies = directory.join("deps");
	let output = directory.join(format!("recipe-script-{}{}", std::process::id(), std::env::consts::EXE_SUFFIX));
	fs::metadata(&library).unwrap_or_else(|error| panic!("cannot inspect {}: {error}", library.display()));
	let status = compile_script(source, &library, &dependencies, &output);
	if !status.success() {
		fs::remove_file(&output).ok();
		std::process::exit(status.code().unwrap_or(1));
	}
	for (key, value) in settings { unsafe { std::env::set_var(key, value); } }
	let inherited_device = std::env::var("RECIPE_DEVICE").ok();
	if let Some(selection) = device.or(inherited_device.as_deref()) {
		match recipe::run_remote_script(&output, selection, config, arguments) {
			Ok(Some(status)) => {
				fs::remove_file(&output).ok();
				std::process::exit(status.code().unwrap_or(1));
			}
			Ok(None) => {}
			Err(error) => { fs::remove_file(&output).ok(); invalid(&error.to_string()); }
		}
	}
	let mut command = Command::new(&output);
	command.args(arguments);
	command.env("RECIPE_BINARY", std::env::current_exe().expect("cannot locate recipe"));
	if let Some(device) = device {
		command.env("RECIPE_DEVICE", device);
	}
	if let Some(config) = config {
		command.env("RECIPE_CONFIG", config);
	}
	#[cfg(unix)]
	{
		extern "C" fn wait_for_script(_: i32) {}
		unsafe extern "C" { fn signal(number: i32, handler: extern "C" fn(i32)) -> usize; }
		unsafe { signal(2, wait_for_script); }
	}
	let status = child_status(&mut command);
	fs::remove_file(&output).ok();
	let status = status.unwrap_or_else(|error| panic!("cannot execute Recipe script: {error}"));
	#[cfg(unix)]
	let code = status.code().unwrap_or_else(|| 128 + std::os::unix::process::ExitStatusExt::signal(&status).unwrap_or(0));
	#[cfg(not(unix))]
	let code = status.code().expect("Recipe script exited without a status code");
	std::process::exit(code);
}

fn main() {
	let mut arguments = std::env::args().skip(1);
	let (mut source, mut device, mut config) = (None::<String>, None::<String>, None::<String>);
	let mut run_seen = false;
	let mut export_seen = false;
	let mut settings = Vec::new();
	let mut script_args = Vec::new();
	while let Some(argument) = arguments.next() {
		if !script_args.is_empty() { script_args.push(argument); continue; }
		if argument == "--" && run_seen && source.is_some() { script_args.extend(arguments); break; }
		if matches!(argument.as_str(), "--help" | "-h") { println!("{USAGE}"); return; }
		if argument == "--ttl" {
			let value = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if value.parse::<u64>().is_err() { invalid("--ttl must be a non-negative whole number of minutes"); }
			if settings.iter().any(|(name, _)| name == "RECIPE_TTL") { invalid("--ttl may be specified only once"); }
			settings.push(("RECIPE_TTL".to_owned(), value));
			continue;
		}
		if matches!(argument.as_str(), "--ctx" | "-p") {
			let value = arguments.next().unwrap_or_else(|| invalid(USAGE));
			let key = if argument == "--ctx" { "RECIPE_CONTEXT" } else { "RECIPE_MESSAGE" };
			if argument == "--ctx" && value.parse::<usize>().ok().is_none_or(|n| n == 0) { invalid("context must be a positive integer"); }
			if settings.iter().any(|(name, _)| name == key) { invalid("run option repeated"); }
			settings.push((key.to_owned(), value));
			continue;
		}
		if source.is_none() && (argument == "stats" || argument == "keys") {
			let path = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if arguments.next().is_some() { invalid(USAGE); }
			(if argument == "stats" { recipe::stats(&path) } else { recipe::keys(&path) }).unwrap_or_else(|error| invalid(&error.to_string()));
			return;
		}
		if source.is_none() && argument == "conventions" {
			let path = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if arguments.next().is_some() { invalid(USAGE); }
			let (report, mismatches) = recipe::conventions(&path).unwrap_or_else(|error| invalid(&error.to_string()));
			print!("{report}");
			std::process::exit(i32::from(mismatches != 0));
		}
		if source.is_none() && argument == "schema" {
			let action = arguments.next().unwrap_or_else(|| invalid(USAGE));
			let path = arguments.next().unwrap_or_else(|| invalid(USAGE));
			match action.as_str() {
				"probe" if arguments.next().is_none() => {
					let packet = recipe::propose_data_schema(&path).unwrap_or_else(|error| invalid(&error.to_string()));
					print!("{packet}");
				}
				"finalize" => {
					let answers = arguments.next().unwrap_or_else(|| invalid(USAGE));
					let output = arguments.next().unwrap_or_else(|| invalid(USAGE));
					if arguments.next().is_some() {
						invalid(USAGE);
					}
					let (schema, decisions) = recipe::finalize_data_schema(&path, &answers).unwrap_or_else(|error| invalid(&error.to_string()));
					fs::write(&output, schema).unwrap_or_else(|error| invalid(&format!("cannot write {output}: {error}")));
					print!("{decisions}");
				}
				_ => invalid(USAGE),
			}
			return;
		}
		if argument == "--device" {
			let selected = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if device.is_some() {
				invalid("--device may be specified only once; use a dot-separated device chain")
			}
			device = Some(selected);
			continue;
		}
		if argument == "--cfg" {
			let selected = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if config.is_some() {
				invalid("--cfg may be specified only once")
			}
			config = Some(selected);
			continue;
		}
		if argument == "run" && source.is_none() {
			if run_seen {
				invalid("run may be specified only once")
			}
			run_seen = true;
			continue;
		}
		if source.is_none() {
			source = Some(argument);
			continue;
		}
		if run_seen { script_args.push(argument); continue; }
		if argument == "export" && !export_seen { export_seen = true; continue; }
		invalid(USAGE)
	}
	if source.is_none() && !run_seen && !export_seen && config.is_none() && settings.is_empty() && let Some(selection) = device.as_deref() {
		let names = recipe::device_names(selection).unwrap_or_else(|error| invalid(&error.to_string()));
		if names.len() != 1 || names[0].contains(':') { invalid("a device worker requires one local device"); }
		recipe::worker_serve(&names[0]).unwrap_or_else(|error| { eprintln!("{error}"); std::process::exit(1) });
		return;
	}
	let source = source.unwrap_or_else(|| invalid(USAGE));
	let devices = device.as_ref().map(|names| recipe::device_names(names).unwrap_or_else(|error| invalid(&error.to_string())));
	let device = device.as_deref();
	let source = Path::new(&source);
	if source.extension().and_then(|value| value.to_str()) != Some("rs") {
		invalid("recipe requires a Rust source")
	}
	if export_seen && devices.as_ref().is_some_and(|names| names.len() != 1) { invalid("export requires one device"); }
	if export_seen { export(source, device) } else { run(source, device, config.as_deref(), &settings, &script_args) }
}
