use std::{fs, path::Path, path::PathBuf, process::Command};

const USAGE: &str = "usage: recipe [run] <source.rs> [--device <[node:]device[.device...]>] [--cfg <precision table>] [--ctx <positions>] [-p <text>] [export]\n\trecipe stats|keys <file.gguf>";

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

fn run(source: &Path, device: Option<&str>, config: Option<&str>, settings: &[(String, String)], arguments: &[String]) {
	let directory = std::env::current_exe().expect("cannot locate recipe").parent().expect("recipe has no parent directory").to_owned();
	let library = library_path(&directory);
	let dependencies = directory.join("deps");
	let output = directory.join(format!("recipe-script-{}{}", std::process::id(), std::env::consts::EXE_SUFFIX));
	fs::metadata(&library).unwrap_or_else(|error| panic!("cannot inspect {}: {error}", library.display()));
	let status = Command::new("rustc")
		.arg("--edition=2024")
		.arg(source)
		.arg("--extern")
		.arg(format!("recipe={}", library.display()))
		.arg("-L")
		.arg(format!("dependency={}", dependencies.display()))
		.arg("-o")
		.arg(&output)
		.status()
		.expect("cannot execute rustc");
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
	let status = command.status();
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
