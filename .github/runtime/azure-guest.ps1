param(
	[Parameter(Mandatory = $true)] [string] $candidateSha,
	[Parameter(Mandatory = $true)] [string] $snapshotSha256,
	[Parameter(Mandatory = $true)] [string] $runtimeSuiteSha256,
	[Parameter(Mandatory = $true)] [string] $snapshotUri,
	[Parameter(Mandatory = $true)] [string] $runtimeSuiteUri
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

function Install-Toolchain {
	param([string] $Root)
	$bootstrap = Join-Path $Root "bootstrap"
	New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
	[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

	Write-Output "== guest: installing the Visual C++ build environment =="
	$vsInstaller = Join-Path $bootstrap "vs_BuildTools-17.14.39.exe"
	Invoke-Download `
		"https://download.visualstudio.microsoft.com/download/pr/fa619120-9c0e-47e6-bfe0-3ee96fb671b2/236367b68ba9a51708263ab10a1c85546cc4a8eca78b365168811d19c4fb2f29/vs_BuildTools.exe" `
		$vsInstaller `
		"236367b68ba9a51708263ab10a1c85546cc4a8eca78b365168811d19c4fb2f29"
	$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
	& $vsInstaller `
		"--quiet" `
		"--wait" `
		"--norestart" `
		"--nocache" `
		"--installPath" `
		$vsRoot `
		"--add" `
		"Microsoft.VisualStudio.Workload.VCTools" `
		"--add" `
		"Microsoft.VisualStudio.Component.VC.Tools.x86.x64" `
		"--add" `
		"Microsoft.VisualStudio.Component.Windows10SDK.20348" `
		"--includeRecommended"
	$vsExit = $LASTEXITCODE
	if (($vsExit -ne 0) -and ($vsExit -ne 3010)) { throw "Visual C++ build environment installation failed with exit code $vsExit" }
	$vsDevCmd = Join-Path $vsRoot "Common7\Tools\VsDevCmd.bat"
	if (![IO.File]::Exists($vsDevCmd)) { throw "Visual C++ developer command file is absent: $vsDevCmd" }
	$vcTools = Join-Path $vsRoot "VC\Tools\MSVC"
	if (![IO.Directory]::Exists($vcTools)) { throw "MSVC tool directory is absent: $vcTools" }
	$sdkHeader = "C:\Program Files (x86)\Windows Kits\10\Include\10.0.20348.0\um\windows.h"
	if (![IO.File]::Exists($sdkHeader)) { throw "Windows SDK header is absent: $sdkHeader" }

	Write-Output "== guest: installing Rust toolchain =="
	$rustup = Join-Path $bootstrap "rustup-init.exe"
	Invoke-WebRequest -UseBasicParsing -Uri "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe" -OutFile $rustup
	Invoke-Native $rustup @("-y", "--profile", "minimal", "--default-toolchain", "stable-x86_64-pc-windows-msvc", "--no-modify-path") "rustup installation"
	$cargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
	$env:Path = "$cargoBin;$env:Path"

	Write-Output "== guest: installing LLVM toolchain =="
	$llvmInstaller = Join-Path $bootstrap "LLVM-18.1.8-win64.exe"
	Invoke-Download `
		"https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.8/LLVM-18.1.8-win64.exe" `
		$llvmInstaller `
		"94af030060d88cc17e9f00ef1663ebdc1126b35e16bebdfa1e807984b70abd8f"
	Invoke-Native $llvmInstaller @("/S") "LLVM installation"

	$clang = Join-Path $env:ProgramFiles "LLVM\bin\clang.exe"
	$linker = Join-Path $env:ProgramFiles "LLVM\bin\lld-link.exe"
	if (![IO.File]::Exists($clang)) { throw "native clang is absent after installation: $clang" }
	if (![IO.File]::Exists($linker)) { throw "native linker is absent after installation: $linker" }

	Write-Output "== guest: installing the CUDA device toolkit =="
	$cudaArchive = Join-Path $bootstrap "cuda_nvcc.zip"
	Invoke-Download `
		"https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.6.85-archive.zip" `
		$cudaArchive `
		"3fb9f76b87c37d02f947354be89b718ad5f2c76b6ab47995265bfa3a068a5e14"
	$cudaStage = Join-Path $bootstrap "cuda-stage"
	if ([IO.Directory]::Exists($cudaStage)) { Remove-Item -Recurse -Force -LiteralPath $cudaStage }
	Expand-Archive -LiteralPath $cudaArchive -DestinationPath $cudaStage -Force
	$cudaPayload = Join-Path $cudaStage "cuda_nvcc-windows-x86_64-12.6.85-archive"
	$cudaRoot = Join-Path $env:ProgramFiles "NVIDIA GPU Computing Toolkit\CUDA\v12.6"
	New-Item -ItemType Directory -Force -Path $cudaRoot | Out-Null
	Copy-Item -Recurse -Force -Path (Join-Path $cudaPayload "*") -Destination $cudaRoot
	[Environment]::SetEnvironmentVariable("CUDA_PATH", $cudaRoot, [EnvironmentVariableTarget]::Machine)
	$env:CUDA_PATH = $cudaRoot
	$env:Path = "$(Join-Path $cudaRoot 'bin');$env:Path"
	$deviceLibrary = Join-Path $cudaRoot "nvvm\libdevice\libdevice.10.bc"
	$nvcc = Join-Path $cudaRoot "bin\nvcc.exe"
	if (![IO.File]::Exists($deviceLibrary)) { throw "CUDA device library is absent after installation: $deviceLibrary" }
	if (![IO.File]::Exists($nvcc)) { throw "CUDA compiler is absent after installation: $nvcc" }

	foreach ($tool in @("rustc", "cargo")) {
		if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is absent after installation" }
	}
	& $clang --version | Select-Object -First 1
	if ($LASTEXITCODE -ne 0) { throw "clang failed after installation with exit code $LASTEXITCODE" }
	& $linker --version | Select-Object -First 1
	if ($LASTEXITCODE -ne 0) { throw "lld-link failed after installation with exit code $LASTEXITCODE" }
	Invoke-Native "rustc" @("--version") "rustc"
	Invoke-Native "cargo" @("--version") "cargo"
	Invoke-Native $nvcc @("--version") "nvcc"
	Write-Output "toolchain ready clang=$clang linker=$linker cuda=$cudaRoot vs=$vsRoot"
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
		if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is absent after entering the Visual C++ developer environment" }
	}
	& cl.exe /Bv 2>&1 | Select-Object -First 3
	if ($LASTEXITCODE -ne 0) { throw "cl.exe failed after entering the Visual C++ developer environment with exit code $LASTEXITCODE" }
}

try {
	$root = "C:\recipe"
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

	Write-Output "== guest: GPU and driver =="
	$smi = "C:\Windows\System32\nvidia-smi.exe"
	if (![IO.File]::Exists($smi)) { throw "nvidia-smi is absent: the GPU driver extension did not install" }
	Invoke-Native $smi @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader") "nvidia-smi"
	$gpu = (& $smi --query-gpu=name --format=csv,noheader) -join ""
	if ($gpu -notmatch "T4") { throw "the allocated GPU is not a T4: $gpu" }

	Install-Toolchain -Root $root
	$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
	Enter-VsDeveloperEnvironment -VsRoot $vsRoot
	$clang = Join-Path $env:ProgramFiles "LLVM\bin\clang.exe"

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
	& (Join-Path $work "target\release\recipe.exe") "--device" "nv0" (Join-Path $runtime "suite.rs") 2>&1 | Tee-Object -FilePath (Join-Path $work "run.log")
	if ($LASTEXITCODE -ne 0) { throw "the runtime suite failed with exit code $LASTEXITCODE" }
	Pop-Location

	$log = Get-Content -Raw -LiteralPath (Join-Path $work "run.log")
	Write-Output $log
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
