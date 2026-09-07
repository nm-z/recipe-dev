param(
	[Parameter(Mandatory = $true)] [string] $testSha,
	[Parameter(Mandatory = $true)] [string] $snapshotSha256
)

# Runs inside the Windows GPU worker, invoked through managed Run Command. It
# holds no cloud-management credential: everything it needs arrived with the
# snapshot.
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

try {
	$root = "C:\recipe"
	# One directory per candidate: two commits must never share a mutable tree.
	$work = Join-Path $root $testSha
	if (Test-Path -LiteralPath $work) { Remove-Item -Recurse -Force -LiteralPath $work }
	New-Item -ItemType Directory -Force -Path $work | Out-Null

	Write-Output "== guest: verifying the snapshot =="
	$archive = Join-Path $root "snapshot.tar.gz"
	[IO.File]::WriteAllBytes($archive, [Convert]::FromBase64String((Get-Content -Raw -LiteralPath (Join-Path $root "snapshot.b64"))))
	$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower()
	if ($actual -ne $snapshotSha256.ToLower()) { throw "snapshot checksum mismatch: $actual != $snapshotSha256" }
	Write-Output "snapshot verified sha256=$actual"
	Invoke-Native "tar.exe" @("-xzf", $archive, "-C", $work) "snapshot extraction"

	Write-Output "== guest: GPU and driver =="
	$smi = "C:\Windows\System32\nvidia-smi.exe"
	if (!(Test-Path -LiteralPath $smi)) { throw "nvidia-smi is absent: the GPU driver extension did not install" }
	Invoke-Native $smi @("--query-gpu=name,driver_version,memory.total", "--format=csv,noheader") "nvidia-smi"
	$gpu = (& $smi --query-gpu=name --format=csv,noheader) -join ""
	if ($gpu -notmatch "T4") { throw "the allocated GPU is not a T4: $gpu" }

	Write-Output "== guest: toolchain =="
	foreach ($tool in @("rustc", "cargo")) {
		if (!(Get-Command $tool -ErrorAction SilentlyContinue)) { throw "$tool is absent from the worker image" }
	}
	Invoke-Native "rustc" @("--version") "rustc"
	Invoke-Native "cargo" @("--version") "cargo"
	$clang = Join-Path $env:ProgramFiles "LLVM\bin\clang.exe"
	if (!(Test-Path -LiteralPath $clang -PathType Leaf)) { throw "native clang is absent: $clang" }
	if (-not $env:CUDA_PATH) { throw "CUDA_PATH is not set: the CUDA toolkit is required for Recipe's NVIDIA backend" }
	Write-Output "CUDA_PATH=$env:CUDA_PATH"

	Write-Output "== guest: building with the NVIDIA backend =="
	Push-Location $work
	Invoke-Native "cargo" @("build", "--release", "--lib", "--bin", "recipe") "the native GPU build"

	Write-Output "== guest: executing the suite on nv0 =="
	New-Item -ItemType Directory -Force -Path (Join-Path $work "evidence"), (Join-Path $work "gpu-work") | Out-Null
	$env:RECIPE_SUITE_ROOT = Join-Path $work ".github\runtime"
	$env:RECIPE_SUITE_WORK = Join-Path $work "gpu-work"
	$env:RECIPE_EVIDENCE = Join-Path $work "evidence\suite.json"
	# --device nv0 hard-errors when the device is absent; a build carrying the
	# nvidia cfg does not add a CPU device, so there is no silent fallback.
	& (Join-Path $work "target\release\recipe.exe") "--device" "nv0" ".github\runtime\suite.rs" 2>&1 | Tee-Object -FilePath (Join-Path $work "run.log")
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
