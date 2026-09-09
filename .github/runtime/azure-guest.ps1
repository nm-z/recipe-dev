param(
	[Parameter(Mandatory = $true)] [ValidateSet("preflight", "execute")] [string] $phase,
	[Parameter(Mandatory = $true)] [string] $candidateSha,
	[Parameter(Mandatory = $true)] [string] $snapshotSha256,
	[Parameter(Mandatory = $true)] [string] $runtimeSuiteSha256,
	[Parameter(Mandatory = $true)] [string] $snapshotUriEncoded,
	[Parameter(Mandatory = $true)] [string] $runtimeSuiteUriEncoded
)

# Runs inside the Windows GPU worker, invoked through managed Run Command. It
# holds no cloud-management credential. The archive URLs are short-lived and
# read-only.
#
# Every native command's exit code is checked immediately, so a later
# successful command can never mask an earlier failing executable.

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Invoke-Native {
	param([string] $Path, [string[]] $Arguments, [string] $What)
	& $Path @Arguments
	if ($LASTEXITCODE -ne 0) { throw "$What failed with exit code $LASTEXITCODE" }
}

function Invoke-Download {
	param([string] $Uri, [string] $Destination, [string] $Sha256)
	Write-Output "downloading payload to $Destination"
	try {
		Invoke-WebRequest -UseBasicParsing -Uri $Uri -OutFile $Destination
	} catch {
		throw "protected download failed for $Destination"
	}
	$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $Destination).Hash.ToLowerInvariant()
	if ($actual -ne $Sha256.ToLowerInvariant()) { throw "download checksum mismatch for ${Destination}: $actual != $Sha256" }
}

function Convert-EncodedUri {
	param([string] $Encoded)
	if ($Encoded -notmatch '^[A-Za-z0-9_-]+$') { throw "protected URL encoding contains an invalid character" }
	$base64 = $Encoded.Replace("-", "+").Replace("_", "/")
	switch ($base64.Length % 4) {
		0 { }
		2 { $base64 += "==" }
		3 { $base64 += "=" }
		default { throw "protected URL encoding has an invalid length" }
	}
	return [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($base64))
}

function Enter-VsDeveloperEnvironment {
	param([string] $VsRoot)
	$vsDevCmd = Join-Path $VsRoot "Common7\Tools\VsDevCmd.bat"
	if (![IO.File]::Exists($vsDevCmd)) { throw "Visual C++ developer command file is absent: $vsDevCmd" }
	Write-Output "== guest: entering the Visual C++ developer environment =="
	$environment = & cmd.exe /d /s /c "call `"$vsDevCmd`" -arch=amd64 -host_arch=amd64 && set" 2>&1
	if ($LASTEXITCODE -ne 0) { throw "Visual C++ developer environment failed with exit code $LASTEXITCODE" }
	foreach ($line in $environment) {
		$entry = [string]$line
		$separator = $entry.IndexOf("=")
		if ($separator -le 0) { continue }
		$name = $entry.Substring(0, $separator)
		$value = $entry.Substring($separator + 1)
		[Environment]::SetEnvironmentVariable($name, $value, [EnvironmentVariableTarget]::Process)
	}
	$env:Path = [Environment]::GetEnvironmentVariable("Path", [EnvironmentVariableTarget]::Process)
	foreach ($tool in @("cl.exe", "link.exe", "lib.exe", "rc.exe")) {
		$command = Get-Command $tool -ErrorAction SilentlyContinue
		if (-not $command) { throw "$tool is absent after entering the Visual C++ developer environment" }
		Write-Output "$tool=$($command.Source)"
	}
}

function Resolve-VsRoot {
	$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
	if (![IO.File]::Exists($vswhere)) { throw "Visual Studio discovery is absent: $vswhere" }
	$roots = @(& $vswhere "-latest" "-products" "*" "-requires" "Microsoft.VisualStudio.Component.VC.Tools.x86.x64" "-property" "installationPath")
	if ($LASTEXITCODE -ne 0) { throw "Visual Studio discovery failed with exit code $LASTEXITCODE" }
	$roots = @($roots | Where-Object { [IO.Directory]::Exists($_) })
	if ($roots.Count -eq 0) { throw "the DSVM image has no Visual Studio instance with x64 C++ tools" }
	return [string]$roots[0]
}

function Resolve-CudaRoot {
	$candidates = [Collections.Generic.List[string]]::new()
	foreach ($root in @(
		$env:CUDA_PATH,
		[Environment]::GetEnvironmentVariable("CUDA_PATH", [EnvironmentVariableTarget]::Machine)
	)) {
		if ($root) { $candidates.Add($root) }
	}
	$base = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA"
	if ([IO.Directory]::Exists($base)) {
		foreach ($directory in @(Get-ChildItem -LiteralPath $base -Directory | Sort-Object { [Version]($_.Name.TrimStart("v")) } -Descending)) {
			$candidates.Add($directory.FullName)
		}
	}
	$seen = @{}
	foreach ($candidate in $candidates) {
		$root = [IO.Path]::GetFullPath($candidate)
		if ($seen.ContainsKey($root)) { continue }
		$seen[$root] = $true
		$deviceLibrary = Join-Path $root "nvvm\libdevice\libdevice.10.bc"
		$nvcc = Join-Path $root "bin\nvcc.exe"
		if ([IO.File]::Exists($deviceLibrary) -and [IO.File]::Exists($nvcc)) { return $root }
	}
	throw "the DSVM image has no complete CUDA root with nvcc and libdevice"
}

function Confirm-Gpu {
	Write-Output "== guest: GPU and driver =="
	$smiCandidates = @(
		"C:\Windows\System32\nvidia-smi.exe",
		(Join-Path $env:ProgramFiles "NVIDIA Corporation\NVSMI\nvidia-smi.exe")
	)
	$script:Smi = $null
	foreach ($candidate in $smiCandidates) {
		if ([IO.File]::Exists($candidate)) { $script:Smi = $candidate; break }
	}
	if (-not $script:Smi) {
		$command = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
		if ($command) { $script:Smi = $command.Source }
	}
	if (-not $script:Smi) { throw "nvidia-smi is absent from the DSVM image" }
	$nvcuda = "C:\Windows\System32\nvcuda.dll"
	if (![IO.File]::Exists($nvcuda)) { throw "the NVIDIA runtime library is absent: $nvcuda" }
	Invoke-Native $script:Smi @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader") "nvidia-smi"
	$script:Gpu = (& $script:Smi --query-gpu=name --format=csv,noheader) -join ""
	if ($LASTEXITCODE -ne 0) { throw "GPU identity query failed with exit code $LASTEXITCODE" }
	if ($script:Gpu -notmatch "T4") { throw "the allocated GPU is not a T4: $script:Gpu" }
}

function Initialize-Toolchain {
	param([string] $Root, [bool] $AllowInstall)
	$bootstrap = Join-Path $Root "bootstrap"
	New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
	[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

	$script:VsRoot = Resolve-VsRoot
	Enter-VsDeveloperEnvironment -VsRoot $script:VsRoot
	$script:CudaRoot = Resolve-CudaRoot
	$env:CUDA_PATH = $script:CudaRoot
	$env:Path = "$(Join-Path $script:CudaRoot 'bin');$env:Path"

	$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
	if ([IO.Directory]::Exists($cargoBin)) { $env:Path = "$cargoBin;$env:Path" }
	$rustReady = $false
	if ((Get-Command "rustc" -ErrorAction SilentlyContinue) -and (Get-Command "cargo" -ErrorAction SilentlyContinue)) {
		$rustDetails = (& rustc -vV) -join "`n"
		$rustReady = ($LASTEXITCODE -eq 0) -and ($rustDetails -match '(?m)^host: .*windows-msvc$')
	}
	if (!$rustReady) {
		if (!$AllowInstall) {
			Write-Output "rust=install-required"
		} else {
			Write-Output "== guest: installing Rust =="
			$rustup = Join-Path $bootstrap "rustup-init.exe"
			Invoke-Download `
				"https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-pc-windows-msvc/rustup-init.exe" `
				$rustup `
				"88d8258dcf6ae4f7a80c7d1088e1f36fa7025a1cfd1343731b4ee6f385121fc0"
			Invoke-Native $rustup @("-y", "--profile", "minimal", "--default-toolchain", "stable-x86_64-pc-windows-msvc", "--no-modify-path") "rustup installation"
			$env:Path = "$cargoBin;$env:Path"
			$rustReady = $true
		}
	}

	$script:Clang = Join-Path $env:ProgramFiles "LLVM\bin\clang.exe"
	$script:Linker = Join-Path $env:ProgramFiles "LLVM\bin\lld-link.exe"
	$llvmReady = [IO.File]::Exists($script:Clang) -and [IO.File]::Exists($script:Linker)
	if (!$llvmReady) {
		if (!$AllowInstall) {
			Write-Output "llvm=install-required"
		} else {
			Write-Output "== guest: installing LLVM =="
			$llvmInstaller = Join-Path $bootstrap "LLVM-22.1.0-win64.exe"
			Invoke-Download `
				"https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.0/LLVM-22.1.0-win64.exe" `
				$llvmInstaller `
				"b31d5f54942e017cb878e594529723dd629cc7b54c9bf7a331e2dc01e8ea5e75"
			$llvmProcess = Start-Process -FilePath $llvmInstaller -ArgumentList "/S" -Wait -PassThru
			if ($llvmProcess.ExitCode -ne 0) { throw "LLVM installation failed with exit code $($llvmProcess.ExitCode)" }
			foreach ($attempt in 1..90) {
				$llvmReady = [IO.File]::Exists($script:Clang) -and [IO.File]::Exists($script:Linker)
				if ($llvmReady) { break }
				Start-Sleep -Seconds 1
			}
		}
	}

	if ($rustReady) {
		$rustDetails = (& rustc -vV) -join "`n"
		if ($LASTEXITCODE -ne 0) { throw "rustc failed with exit code $LASTEXITCODE" }
		if ($rustDetails -notmatch '(?m)^host: .*windows-msvc$') { throw "rustc is not an MSVC host: $rustDetails" }
		Invoke-Native "cargo" @("--version") "cargo"
	}
	if ($llvmReady) {
		if (![IO.File]::Exists($script:Clang)) { throw "native clang is absent: $script:Clang" }
		if (![IO.File]::Exists($script:Linker)) { throw "native linker is absent: $script:Linker" }
		Invoke-Native $script:Clang @("--version") "clang"
		Invoke-Native $script:Linker @("--version") "lld-link"
	}
	$nvcc = Join-Path $script:CudaRoot "bin\nvcc.exe"
	Invoke-Native $nvcc @("--version") "nvcc"
	$state = if ($AllowInstall) { "toolchain ready" } else { "platform ready" }
	Write-Output "$state cuda=$script:CudaRoot vs=$script:VsRoot"
}

try {
	$root = "C:\recipe"
	Confirm-Gpu
	Initialize-Toolchain -Root $root -AllowInstall ($phase -eq "execute")
	if ($phase -eq "preflight") {
		Write-Output "PREFLIGHT EXIT 0"
		exit 0
	}
	$snapshotUri = Convert-EncodedUri $snapshotUriEncoded
	$runtimeSuiteUri = Convert-EncodedUri $runtimeSuiteUriEncoded

	# One directory per candidate: two commits must never share a mutable tree.
	$work = Join-Path $root $candidateSha
	if ([IO.Directory]::Exists($work)) { Remove-Item -Recurse -Force -LiteralPath $work }
	New-Item -ItemType Directory -Force -Path $work | Out-Null

	Write-Output "== guest: verifying the snapshot =="
	$archive = Join-Path $root "snapshot.tar.gz"
	Invoke-Download $snapshotUri $archive $snapshotSha256
	$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower()
	Write-Output "snapshot verified sha256=$actual"
	Invoke-Native "tar.exe" @("-xzf", $archive, "-C", $work) "snapshot extraction"

	$runtimeArchive = Join-Path $root "runtime-suite.tar.gz"
	$runtime = Join-Path $root "trusted-runtime"
	Invoke-Download $runtimeSuiteUri $runtimeArchive $runtimeSuiteSha256
	$runtimeActual = (Get-FileHash -Algorithm SHA256 -LiteralPath $runtimeArchive).Hash.ToLower()
	if ([IO.Directory]::Exists($runtime)) { Remove-Item -Recurse -Force -LiteralPath $runtime }
	New-Item -ItemType Directory -Force -Path $runtime | Out-Null
	Invoke-Native "tar.exe" @("-xzf", $runtimeArchive, "-C", $runtime) "trusted runtime extraction"
	Write-Output "trusted runtime verified sha256=$runtimeActual"

	Write-Output "== guest: building with the NVIDIA backend =="
	Push-Location $work
	Invoke-Native "cargo" @("build", "--release", "--lib", "--bin", "recipe") "the native GPU build"

	Write-Output "== guest: executing the suite on nv0 =="
	New-Item -ItemType Directory -Force -Path (Join-Path $work "evidence"), (Join-Path $work "gpu-work") | Out-Null
	$env:RECIPE_SUITE_ROOT = $runtime
	$env:RECIPE_SUITE_WORK = Join-Path $work "gpu-work"
	$env:RECIPE_EVIDENCE = Join-Path $work "evidence\suite.json"
	# --device nv0 hard-errors when the device is absent; a build carrying the
	# nvidia cfg does not add a CPU device, so there is no silent fallback.
	$runStdout = Join-Path $work "run.stdout.log"
	$runStderr = Join-Path $work "run.stderr.log"
	$runProcess = Start-Process `
		-FilePath (Join-Path $work "target\release\recipe.exe") `
		-ArgumentList @("--device", "nv0", (Join-Path $runtime "suite.rs")) `
		-Wait -PassThru `
		-RedirectStandardOutput $runStdout `
		-RedirectStandardError $runStderr
	$log = ((Get-Content -Raw -LiteralPath $runStdout), (Get-Content -Raw -LiteralPath $runStderr)) -join "`n"
	[IO.File]::WriteAllText((Join-Path $work "run.log"), $log, [Text.UTF8Encoding]::new($false))
	Write-Output $log
	if ($runProcess.ExitCode -ne 0) { throw "the runtime suite failed with exit code $($runProcess.ExitCode)" }
	Pop-Location

	if ($log -notmatch "SUITE PASS") { throw "the suite did not report SUITE PASS" }
	$route = ([regex]::Match($log, '(?m)^selected route (\S+)')).Groups[1].Value
	if (-not $route) { throw "no route line: the suite did not dispatch" }
	$device = $route.Split(':')[-1]
	if ($device -notlike "nv*") { throw "expected an nv device, got '$device'; CPU fallback is a failure" }
	Write-Output "executed on $route"

	Write-Output "SUITE-EVIDENCE-BEGIN"
	Write-Output (Get-Content -Raw -LiteralPath (Join-Path $work "evidence\suite.json"))
	Write-Output "SUITE-EVIDENCE-END"
	Write-Output "GUEST EXIT 0"
} catch {
	Write-Output "GUEST FAILURE: $_"
	Write-Output ($_.ScriptStackTrace)
	exit 1
}
