use std::{fs, path::Path, path::PathBuf, process::Command};

const USAGE: &str = "usage: recipe [--device <name>]... <source.rs> [export]\n       recipe --worker <device>";

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

fn run(source: &Path, device: Option<&str>) {
	let directory = std::env::current_exe().expect("cannot locate recipe").parent().expect("recipe has no parent directory").to_owned();
	let library = library_path(&directory);
	let dependencies = directory.join("deps");
	// Each invocation compiles to its own output, so concurrent invocations never share one.
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
	let mut command = Command::new(&output);
	command.env("RECIPE_BINARY", std::env::current_exe().expect("cannot locate recipe"));
	if let Some(device) = device {
		command.env("RECIPE_DEVICE", device);
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
	let (mut source, mut operation, mut device) = (None::<String>, None::<String>, None::<String>);
	while let Some(argument) = arguments.next() {
		if argument == "--worker" {
			let name = arguments.next().unwrap_or_else(|| invalid("--worker requires a device name"));
			recipe::worker_serve(&name).unwrap_or_else(|error| {
				eprintln!("{error}");
				std::process::exit(1)
			});
			return;
		}
		if argument == "--device" {
			let selected = arguments.next().unwrap_or_else(|| invalid(USAGE));
			if let Some(devices) = &mut device {
				devices.push(',');
				devices.push_str(&selected);
			} else {
				device = Some(selected);
			}
			continue;
		}
		if argument.starts_with("--") {
			invalid(USAGE)
		}
		if source.is_none() {
			source = Some(argument);
			continue;
		}
		if operation.is_none() {
			operation = Some(argument);
			continue;
		}
		invalid(USAGE)
	}
	let source = source.unwrap_or_else(|| invalid(USAGE));
	let device = device.as_deref();
	let source = Path::new(&source);
	if source.extension().and_then(|value| value.to_str()) != Some("rs") {
		invalid("recipe requires a Rust source")
	}
	match operation.as_deref() {
		None => run(source, device),
		Some("export") if device.is_some_and(|names| names.contains(',')) => invalid("export requires one device"),
		Some("export") => export(source, device),
		Some(_) => invalid(USAGE),
	}
}
