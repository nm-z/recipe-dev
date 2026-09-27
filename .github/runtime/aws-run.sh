#!/usr/bin/env bash
# Runs the immutable candidate archive on one AWS EC2 NVIDIA L4 instance.
#
# The AWS credential stays in this controller process. The instance receives
# only presigned URLs for the archive, the worker program and its log, and fixed
# digest and commit values. It terminates itself when the worker ends, and
# aws-terminate.sh terminates it and removes the run's objects regardless.
set -euo pipefail

: "${AWS_ACCESS_KEY_ID:?AWS credentials are required}"
: "${AWS_SECRET_ACCESS_KEY:?AWS credentials are required}"
: "${AWS_REGION:?AWS_REGION is required}"
: "${AWS_BUCKET:?AWS_BUCKET is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${TRUSTED_RUNTIME:?TRUSTED_RUNTIME is required}"
# The worker runs one workload: the suite (default), or the composition harness over
# RECIPE_TRIAL_COUNT cursors from RECIPE_TRIAL_CURSOR, whose stderr packets are the evidence.
RECIPE_WORKLOAD="${RECIPE_WORKLOAD:-suite}"
case "$RECIPE_WORKLOAD" in
	suite) ;;
	trial)
		: "${RECIPE_TRIAL_CURSOR:?RECIPE_TRIAL_CURSOR is required}"
		: "${RECIPE_TRIAL_COUNT:?RECIPE_TRIAL_COUNT is required}"
		case "$RECIPE_TRIAL_CURSOR$RECIPE_TRIAL_COUNT" in
			''|*[!0-9]*) echo "the trial cursor and count must be integers" >&2; exit 1 ;;
		esac
		;;
	*) echo "RECIPE_WORKLOAD must be suite or trial" >&2; exit 1 ;;
esac

# g6.xlarge carries one NVIDIA L4. The Deep Learning Base AMI provides the NVIDIA
# driver and the CUDA toolkit the worker compiles against.
INSTANCE_TYPE="${INSTANCE_TYPE:-g6.xlarge}"
AMI_PARAMETER="${AMI_PARAMETER:-/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id}"
ROOT_VOLUME_GB="${ROOT_VOLUME_GB:-100}"
# The worker's build and its workload each have their own hard timeout; the
# instance powers off at INSTANCE_LIFETIME_MINUTES whatever the worker is doing,
# and the presigned URLs outlive it. Both stay inside the controller's session.
WORKER_EXECUTION_TIMEOUT_SECONDS="${WORKER_EXECUTION_TIMEOUT_SECONDS:-1500}"
INSTANCE_LIFETIME_MINUTES="${INSTANCE_LIFETIME_MINUTES:-80}"
URL_EXPIRY_SECONDS="${URL_EXPIRY_SECONDS:-6000}"
BOOT_DEADLINE_SECONDS="${BOOT_DEADLINE_SECONDS:-600}"
RUN_DEADLINE_SECONDS="${RUN_DEADLINE_SECONDS:-4500}"
POLL_SECONDS="${POLL_SECONDS:-20}"

for value in "$WORKER_EXECUTION_TIMEOUT_SECONDS" "$INSTANCE_LIFETIME_MINUTES" "$URL_EXPIRY_SECONDS" "$ROOT_VOLUME_GB"; do
	case "$value" in
		''|*[!0-9]*) echo "AWS runner limits must be integers" >&2; exit 1 ;;
	esac
done
case "$AWS_BUCKET" in
	''|*[!a-z0-9.-]*) echo "AWS_BUCKET must be a plain S3 bucket name" >&2; exit 1 ;;
esac

mkdir -p evidence
command -v aws >/dev/null 2>&1 || { echo "the AWS CLI is unavailable" >&2; exit 1; }
aws --version

# Writes evidence/blocker.json and stops the run with its detail.
blocker() {
	jq -n --arg blocker "$1" --arg detail "$2" --arg resolution "$3" '{ blocker: $blocker, detail: $detail, resolution: $resolution }' > evidence/blocker.json
	cat evidence/blocker.json
	echo "recipe/linux-gpu is blocked: $2" >&2
	exit 1
}

# Presigns an S3 PUT with SigV4 query authentication; `aws s3 presign` covers GET only.
hmac_hex() {
	printf '%s' "$2" | openssl dgst -sha256 -mac HMAC -macopt "$1" | awk '{ print $NF }'
}
presign_put() {
	local key="$1" host="$AWS_BUCKET.s3.$AWS_REGION.amazonaws.com"
	local amz_date date scope query canonical string_to_sign signing_key
	amz_date="$(date -u +%Y%m%dT%H%M%SZ)"
	date="${amz_date:0:8}"
	scope="$date/$AWS_REGION/s3/aws4_request"
	query="X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=$(jq -rn --arg v "$AWS_ACCESS_KEY_ID/$scope" '$v | @uri')&X-Amz-Date=$amz_date&X-Amz-Expires=$URL_EXPIRY_SECONDS"
	if [ -n "${AWS_SESSION_TOKEN:-}" ]; then
		query="$query&X-Amz-Security-Token=$(jq -rn --arg v "$AWS_SESSION_TOKEN" '$v | @uri')"
	fi
	query="$query&X-Amz-SignedHeaders=host"
	canonical="$(printf 'PUT\n/%s\n%s\nhost:%s\n\nhost\nUNSIGNED-PAYLOAD' "$key" "$query" "$host")"
	string_to_sign="$(printf 'AWS4-HMAC-SHA256\n%s\n%s\n%s' "$amz_date" "$scope" "$(printf '%s' "$canonical" | sha256sum | awk '{ print $1 }')")"
	signing_key="$(hmac_hex "key:AWS4$AWS_SECRET_ACCESS_KEY" "$date")"
	signing_key="$(hmac_hex "hexkey:$signing_key" "$AWS_REGION")"
	signing_key="$(hmac_hex "hexkey:$signing_key" s3)"
	signing_key="$(hmac_hex "hexkey:$signing_key" aws4_request)"
	printf 'https://%s/%s?%s&X-Amz-Signature=%s\n' "$host" "$key" "$query" "$(hmac_hex "hexkey:$signing_key" "$string_to_sign")"
}

echo "== resolving the AWS account =="
if ! identity="$(aws sts get-caller-identity --output json 2>&1)"; then
	printf '%s\n' "$identity" >&2
	blocker aws-credential-rejected "AWS rejected the controller credential." "Check the AWS_ROLE_ARN secret and the role's trust policy for this repository."
fi
printf '%s' "$identity" | jq -r '"account \(.Account) as \(.Arn)"'

echo "== verifying the immutable archive =="
actual_sha256="$(sha256sum "$SNAPSHOT" | awk '{print $1}')"
if [ "$actual_sha256" != "$SNAPSHOT_SHA256" ]; then
	echo "snapshot checksum mismatch before upload: $actual_sha256 != $SNAPSHOT_SHA256" >&2
	exit 1
fi
if [ "$RECIPE_WORKLOAD" = trial ]; then
	[ -f "$TRUSTED_RUNTIME/harness.rs" ] || { echo "the trial harness is absent" >&2; exit 1; }
	tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" harness.rs
else
	[ -f "$TRUSTED_RUNTIME/suite.rs" ] || { echo "trusted suite is absent" >&2; exit 1; }
	[ -d "$TRUSTED_RUNTIME/data" ] || { echo "trusted suite data is absent" >&2; exit 1; }
	tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" suite.rs data
fi

request_key="${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-1}-${CANDIDATE_SHA:0:12}"
s3_prefix="recipe-${RECIPE_WORKLOAD/suite/runtime}/${request_key}"
printf '%s\n' "$s3_prefix" > aws-s3-prefix

cat > worker.sh <<'WORKER'
#!/usr/bin/env bash
set -euo pipefail

: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${WORKER_EXECUTION_TIMEOUT_SECONDS:?worker execution timeout is required}"

source_root="$(pwd)"
root="${TMPDIR:-/tmp}/recipe-aws-${CANDIDATE_SHA:0:12}-$$"
mkdir -p "$root"
cp "$source_root/recipe-source.tar.gz" "$root/recipe-source.tar.gz"
cp "$source_root/trusted-runtime.tar.gz" "$root/trusted-runtime.tar.gz"
cd "$root"
archive="$root/recipe-source.tar.gz"
archive_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [ "$archive_sha256" != "$SNAPSHOT_SHA256" ]; then
	echo "snapshot checksum mismatch in the AWS worker: $archive_sha256 != $SNAPSHOT_SHA256" >&2
	exit 1
fi

echo "== worker: GPU information =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if ! nvidia-smi --query-gpu=name --format=csv,noheader | awk 'BEGIN { IGNORECASE=1 } /L4/ { found=1 } END { exit(found ? 0 : 1) }'; then
	echo "the allocated GPU is not an NVIDIA L4" >&2
	exit 1
fi

echo "== worker: native Ubuntu userspace =="
work="$root/work"
toolchains="$root/toolchains"
rustup_home="$toolchains/rustup"
cargo_home="$toolchains/cargo"
llvm_version="23.1.0"
llvm_archive="LLVM-${llvm_version}-Linux-X64.tar.xz"
llvm_sha256="18da30f77f475688a18f7704d23f9f155ae007ed9922dbed6850a9419d9fec8c"
llvm_root="$toolchains/LLVM-${llvm_version}-Linux-X64"
mkdir -p "$work" "$toolchains"
tar -xzf "$archive" -C "$work"
mkdir -p "$work/.github/runtime"
tar -xzf "$root/trusted-runtime.tar.gz" -C "$work/.github/runtime"

echo "== worker: install user-local Rust =="
export RUSTUP_HOME="$rustup_home"
export CARGO_HOME="$cargo_home"
export PATH="$CARGO_HOME/bin:$PATH"
curl -fsSL https://sh.rustup.rs -o "$toolchains/rustup-init.sh"
sh "$toolchains/rustup-init.sh" -y --profile minimal --default-toolchain stable

echo "== worker: install user-local LLVM =="
curl -fsSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvm_version}/$llvm_archive" -o "$toolchains/$llvm_archive"
printf '%s  %s\n' "$llvm_sha256" "$toolchains/$llvm_archive" | sha256sum -c -
tar -xJf "$toolchains/$llvm_archive" -C "$toolchains"
[ -x "$llvm_root/bin/clang" ] || { echo "user-local clang is absent" >&2; exit 1; }
[ -x "$llvm_root/bin/ld.lld" ] || { echo "user-local ld.lld is absent" >&2; exit 1; }

nvcc_path="$(command -v nvcc || true)"
[ -n "$nvcc_path" ] || { echo "nvcc is absent from the AWS worker" >&2; exit 1; }
nvcc_path="$(readlink -f "$nvcc_path")"
cuda_root="$(cd "$(dirname "$nvcc_path")/.." && pwd -P)"
cuda_device_library="$cuda_root/nvvm/libdevice/libdevice.10.bc"
[ -f "$cuda_device_library" ] || { echo "CUDA libdevice is absent: $cuda_device_library" >&2; exit 1; }
ldconfig -p | grep -q 'libcuda\.so\.1' || { echo "libcuda.so.1 is absent from the AWS worker" >&2; exit 1; }

echo "== worker: patch extracted platform paths =="
candidate_manifest="$work/Cargo.toml"
[ -f "$candidate_manifest" ] || { echo "candidate Cargo.toml is absent" >&2; exit 1; }
sed -i \
	-e "s|cpu-compiler = { linux = \"/usr/bin/clang\"|cpu-compiler = { linux = \"$llvm_root/bin/clang\"|" \
	-e "s|cpu-linker = { linux = \"/usr/bin/ld\"|cpu-linker = { linux = \"$llvm_root/bin/ld.lld\"|" \
	-e 's|cpu-linker-driver = { linux = "ld"|cpu-linker-driver = { linux = "lld"|' \
	-e "s|nvidia-compiler = { linux = \"/usr/bin/clang\"|nvidia-compiler = { linux = \"$llvm_root/bin/clang\"|" \
	-e "s|nvidia-toolkit = { linux = \"/opt/cuda\"|nvidia-toolkit = { linux = \"$cuda_root\"|" \
	"$candidate_manifest"
grep -Fq "cpu-compiler = { linux = \"$llvm_root/bin/clang\"" "$candidate_manifest" || { echo "candidate CPU compiler path was not patched" >&2; exit 1; }
grep -Fq "nvidia-compiler = { linux = \"$llvm_root/bin/clang\"" "$candidate_manifest" || { echo "candidate NVIDIA compiler path was not patched" >&2; exit 1; }
grep -Fq "nvidia-toolkit = { linux = \"$cuda_root\"" "$candidate_manifest" || { echo "candidate CUDA root was not patched" >&2; exit 1; }

cd "$work"
echo "== worker: toolchain =="
rustc --version
cargo --version
"$llvm_root/bin/clang" --version | head -1
echo "CUDA root: $cuda_root"
echo "== worker: build with the NVIDIA backend =="
if ! timeout --signal=TERM --kill-after=30s "${WORKER_EXECUTION_TIMEOUT_SECONDS}s" cargo build --release --lib --bin recipe; then
	echo "the AWS worker build failed or reached its hard timeout" >&2
	exit 1
fi
if [ "${RECIPE_WORKLOAD:-suite}" = trial ]; then
	echo "== worker: run the composition harness on nv0, cursor $RECIPE_TRIAL_CURSOR count $RECIPE_TRIAL_COUNT =="
	mkdir -p "$root/evidence" "$work/trial"
	cp "$work/.github/runtime/harness.rs" "$work/harness.rs"
	trial_started="$(date +%s)"
	# The harness prints one composition line per cursor and a failure packet per defect on
	# stderr; that stream is the evidence, so a nonzero exit is recorded rather than fatal.
	set +e
	timeout --signal=TERM --kill-after=30s "${WORKER_EXECUTION_TIMEOUT_SECONDS}s" env \
		RECIPE_DEVICE=nv0 \
		RECIPE_COMPOSITION_RUNNER="$work/target/release/recipe" \
		RECIPE_COMPOSITION_CURSOR="$RECIPE_TRIAL_CURSOR" \
		RECIPE_COMPOSITION_COUNT="$RECIPE_TRIAL_COUNT" \
		RECIPE_COMPOSITION_REPLAY_SEED=17 \
		RECIPE_COMPOSITION_REPRO="$work/trial/repro.rs" \
		RECIPE_TRIAL_DIRECTORY="$work/trial" \
		"$work/target/release/recipe" harness.rs > "$work/trial/harness.out" 2> "$work/trial/harness.log"
	harness_status=$?
	set -e
	echo "== worker: harness exit $harness_status after $(( $(date +%s) - trial_started ))s =="
	nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
	# The snapshot has no history, so the harness base line carries no commit; the candidate stands in.
	awk -v sha="$CANDIDATE_SHA" '{ sub(/^base=commit=[0-9a-f]*/, "base=commit=" sha); print }' "$work/trial/harness.log" > "$root/evidence/trial.log"
	[ -s "$root/evidence/trial.log" ] || { echo "the harness wrote no evidence" >&2; exit 1; }
	printf '%s\n' "RECIPE_TRIAL_LOG_BEGIN"
	base64 -w0 "$root/evidence/trial.log"
	printf '\n%s\n' "RECIPE_TRIAL_LOG_END"
	echo "WORKER EXIT 0"
	exit 0
fi
echo "== worker: execute the suite on nv0 =="
mkdir -p "$work/evidence" "$work/gpu-work"
if ! timeout --signal=TERM --kill-after=30s "${WORKER_EXECUTION_TIMEOUT_SECONDS}s" env \
	RECIPE_SUITE_ROOT="$work/.github/runtime" \
	RECIPE_SUITE_WORK="$work/gpu-work" \
	RECIPE_EVIDENCE="$work/evidence/suite.json" \
	"$work/target/release/recipe" --device nv0 "$work/.github/runtime/suite.rs" 2>&1 | tee "$work/run.log"; then
	echo "the AWS worker suite failed or reached its hard timeout" >&2
	exit 1
fi
mkdir -p "$root/evidence"
cp -r "$work/evidence/." "$root/evidence/"
cp "$work/run.log" "$root/evidence/worker-run.log"

route="$(awk '/^suite device / { print $3; exit }' "$root/evidence/worker-run.log")"
device="${route##*:}"
case "$device" in
	nv*) echo "executed on $route" ;;
	*) echo "expected an NVIDIA device, got '$device'" >&2; exit 1 ;;
esac
[ -f "$root/evidence/suite.json" ] || { echo "the worker returned no suite evidence" >&2; exit 1; }
grep -q "SUITE PASS" "$root/evidence/worker-run.log" || { echo "the suite did not report SUITE PASS" >&2; exit 1; }
printf '%s\n' "RECIPE_SUITE_JSON_BEGIN"
base64 -w0 "$root/evidence/suite.json"
printf '\n%s\n' "RECIPE_SUITE_JSON_END"
echo "WORKER EXIT 0" | tee -a "$root/evidence/worker-run.log"
WORKER
chmod +x worker.sh

echo "== uploading the archive and worker to S3 =="
for file in "$SNAPSHOT" trusted-runtime.tar.gz worker.sh; do
	aws s3 cp --only-show-errors "$file" "s3://$AWS_BUCKET/$s3_prefix/$(basename "$file")"
done
presign_get() {
	aws s3 presign "s3://$AWS_BUCKET/$s3_prefix/$1" --expires-in "$URL_EXPIRY_SECONDS" --region "$AWS_REGION"
}
source_url="$(presign_get recipe-source.tar.gz)"
runtime_url="$(presign_get trusted-runtime.tar.gz)"
worker_url="$(presign_get worker.sh)"
log_url="$(presign_put "$s3_prefix/worker-run.log")"

# A trial carries its own wall-clock budget: toolchain, build and harness together end within it.
TRIAL_BUDGET_SECONDS="${TRIAL_BUDGET_SECONDS:-2400}"
worker_launch="bash worker.sh"
if [ "$RECIPE_WORKLOAD" = trial ]; then
	worker_launch="timeout --signal=TERM --kill-after=30s ${TRIAL_BUDGET_SECONDS}s bash worker.sh"
fi
cat > user-data.sh <<USERDATA
#!/bin/bash
set -uo pipefail
# The instance terminates on power-off, so this bounds its life whatever the worker does.
shutdown -h +$INSTANCE_LIFETIME_MINUTES
export HOME=/root
export PATH=/usr/local/cuda/bin:\$PATH
mkdir -p /opt/recipe-run
cd /opt/recipe-run
{
	curl -fsS --retry 5 -o recipe-source.tar.gz '$source_url' \\
		&& curl -fsS --retry 5 -o trusted-runtime.tar.gz '$runtime_url' \\
		&& curl -fsS --retry 5 -o worker.sh '$worker_url' \\
		&& SNAPSHOT_SHA256=$SNAPSHOT_SHA256 CANDIDATE_SHA=$CANDIDATE_SHA WORKER_EXECUTION_TIMEOUT_SECONDS=$WORKER_EXECUTION_TIMEOUT_SECONDS RECIPE_WORKLOAD=$RECIPE_WORKLOAD RECIPE_TRIAL_CURSOR=${RECIPE_TRIAL_CURSOR:-0} RECIPE_TRIAL_COUNT=${RECIPE_TRIAL_COUNT:-0} $worker_launch
	echo "== worker: exit status \$? =="
} > run.log 2>&1
curl -fsS --retry 5 -X PUT --upload-file run.log '$log_url'
shutdown -h now
USERDATA

echo "== resolving the AMI and subnets =="
ami_id="$(aws ssm get-parameter --region "$AWS_REGION" --name "$AMI_PARAMETER" --query Parameter.Value --output text)"
root_device="$(aws ec2 describe-images --region "$AWS_REGION" --image-ids "$ami_id" --query 'Images[0].RootDeviceName' --output text)"
image_volume_gb="$(aws ec2 describe-images --region "$AWS_REGION" --image-ids "$ami_id" --query "Images[0].BlockDeviceMappings[?DeviceName=='$root_device'].Ebs.VolumeSize | [0]" --output text)"
# The root volume cannot be smaller than the image snapshot.
case "$image_volume_gb" in
	''|*[!0-9]*) ;;
	*) [ "$image_volume_gb" -le "$ROOT_VOLUME_GB" ] || ROOT_VOLUME_GB="$image_volume_gb" ;;
esac
echo "AMI $ami_id root $root_device image ${image_volume_gb}GB volume ${ROOT_VOLUME_GB}GB"
if [ -n "${AWS_SUBNET_ID:-}" ]; then
	subnets="$AWS_SUBNET_ID"
else
	# Every default subnet is a candidate: L4 capacity differs between availability zones.
	subnets="$(aws ec2 describe-subnets --region "$AWS_REGION" --filters Name=default-for-az,Values=true --query 'Subnets[].SubnetId' --output text)"
	[ -n "$subnets" ] || blocker aws-no-subnet "The region $AWS_REGION has no default VPC subnet." "Create a default VPC in $AWS_REGION, or set the AWS_SUBNET_ID repository variable to a subnet with internet access."
fi

echo "== launching the $INSTANCE_TYPE instance =="
instance_id=""
for subnet in $subnets; do
	if launch_output="$(aws ec2 run-instances --region "$AWS_REGION" \
		--image-id "$ami_id" \
		--instance-type "$INSTANCE_TYPE" \
		--count 1 \
		--network-interfaces "DeviceIndex=0,SubnetId=$subnet,AssociatePublicIpAddress=true,DeleteOnTermination=true" \
		--block-device-mappings "DeviceName=$root_device,Ebs={VolumeSize=$ROOT_VOLUME_GB,VolumeType=gp3,DeleteOnTermination=true}" \
		--instance-initiated-shutdown-behavior terminate \
		--metadata-options HttpTokens=required \
		--user-data file://user-data.sh \
		--tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$s3_prefix},{Key=recipe-run,Value=$request_key}]" \
		--query 'Instances[0].InstanceId' --output text 2>&1)"; then
		instance_id="$launch_output"
		break
	fi
	printf '%s\n' "$launch_output" >&2
	case "$launch_output" in
		*InsufficientInstanceCapacity*|*Unsupported*) echo "no $INSTANCE_TYPE capacity in $subnet; trying the next subnet" >&2 ;;
		*VcpuLimitExceeded*) blocker aws-gpu-quota "AWS refused $INSTANCE_TYPE: the account's vCPU quota for on-demand G instances in $AWS_REGION is too low." "Request at least 4 vCPUs for 'Running On-Demand G and VT instances' in Service Quotas for $AWS_REGION, then rerun." ;;
		*PendingVerification*|*OptInRequired*) blocker aws-account-pending "AWS has not finished verifying the account for EC2 in $AWS_REGION." "Wait for AWS to complete account verification, then rerun." ;;
		*UnauthorizedOperation*|*AccessDenied*) blocker aws-permission "The AWS role may not launch EC2 instances." "Allow ec2:RunInstances, ec2:CreateTags, ec2:Describe*, ec2:TerminateInstances, ec2:GetConsoleOutput, ssm:GetParameter and S3 access to the bucket for the AWS_ROLE_ARN role." ;;
		*) exit 1 ;;
	esac
done
[ -n "$instance_id" ] || { echo "no subnet in $AWS_REGION had $INSTANCE_TYPE capacity" >&2; exit 1; }
case "$instance_id" in
	i-*) ;;
	*) echo "AWS did not return an instance ID" >&2; exit 1 ;;
esac
printf '%s\n' "$instance_id" > aws-instance-id
echo "launched $instance_id for $CANDIDATE_SHA"

echo "== polling the instance and its log =="
started="$(date +%s)"
running=false
while :; do
	elapsed=$(($(date +%s) - started))
	if aws s3api head-object --bucket "$AWS_BUCKET" --key "$s3_prefix/worker-run.log" >/dev/null 2>&1; then
		echo "  t=${elapsed}s worker log returned"
		break
	fi
	state="$(aws ec2 describe-instances --region "$AWS_REGION" --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo unknown)"
	echo "  t=${elapsed}s state=$state"
	case "$state" in
		running) running=true ;;
		pending)
			[ "$elapsed" -lt "$BOOT_DEADLINE_SECONDS" ] || { echo "the instance did not start within ${BOOT_DEADLINE_SECONDS}s" >&2; exit 1; }
			;;
		shutting-down|terminated|stopping|stopped)
			# The log upload precedes the power-off; one more look covers the race.
			aws s3api head-object --bucket "$AWS_BUCKET" --key "$s3_prefix/worker-run.log" >/dev/null 2>&1 && break
			echo "the instance reached $state without returning its log" >&2
			exit 1
			;;
	esac
	[ "$elapsed" -lt "$RUN_DEADLINE_SECONDS" ] || { echo "execution deadline of ${RUN_DEADLINE_SECONDS}s exceeded (running=$running)" >&2; exit 1; }
	sleep "$POLL_SECONDS"
done

echo "== collecting the worker evidence =="
aws s3 cp --only-show-errors "s3://$AWS_BUCKET/$s3_prefix/worker-run.log" evidence/worker-run.log
[ -f evidence/worker-run.log ] || { echo "worker log is absent" >&2; exit 1; }
grep -q "WORKER EXIT 0" evidence/worker-run.log || { tail -n 60 evidence/worker-run.log >&2; echo "the worker did not report a zero exit status" >&2; exit 1; }
if [ "$RECIPE_WORKLOAD" = trial ]; then
	# The trial's evidence is the harness stderr: composition lines and failure packets, named
	# the way the issue machine's inbox expects (device prefix before "-run", then the cursor span).
	trial_file="evidence/aws-l4-run-${RECIPE_TRIAL_CURSOR}-$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1)).txt"
	awk '/^RECIPE_TRIAL_LOG_BEGIN$/{capture=1; next} /^RECIPE_TRIAL_LOG_END$/{capture=0; exit} capture{print}' evidence/worker-run.log | tr -d '\r\n' | base64 -d > "$trial_file" || {
		echo "the worker log did not contain the trial log" >&2
		exit 1
	}
	grep -E '^== worker: harness exit' evidence/worker-run.log
	harness_status="$(sed -n 's/^== worker: harness exit \([0-9][0-9]*\).*/\1/p' evidence/worker-run.log | head -1)"
	case "$harness_status" in
		''|*[!0-9]*) echo "the worker log has no valid harness exit status" >&2; exit 1 ;;
	esac
	compositions="$(grep -c '^composition [0-9]*:' "$trial_file" || true)"
	packets="$(grep -c '^RECIPE FAILURE BEGIN$' "$trial_file" || true)"
	echo "compositions: $compositions, packets: $packets, file: $trial_file"
	if [ "$harness_status" -ne 0 ] || [ "$compositions" -ne "$RECIPE_TRIAL_COUNT" ]; then
		echo "the AWS trial returned partial evidence: status=$harness_status compositions=$compositions expected=$RECIPE_TRIAL_COUNT" >&2
		exit 1
	fi
	echo "recipe/aws-trial completed for cursors $RECIPE_TRIAL_CURSOR..$((RECIPE_TRIAL_CURSOR + RECIPE_TRIAL_COUNT - 1))"
	exit 0
fi
awk '/^RECIPE_SUITE_JSON_BEGIN$/{capture=1; next} /^RECIPE_SUITE_JSON_END$/{capture=0; exit} capture{print}' evidence/worker-run.log | tr -d '\r\n' | base64 -d > evidence/suite.json || {
	echo "the worker log did not contain valid suite evidence" >&2
	exit 1
}
[ -s evidence/suite.json ] || { echo "suite evidence is absent" >&2; exit 1; }
route="$(awk '/^suite device / { print $3; exit }' evidence/worker-run.log)"
device="${route##*:}"
case "$device" in
	nv*) ;;
	*) echo "the retrieved evidence does not show an NVIDIA device" >&2; exit 1 ;;
esac
gpu_name="$(awk 'BEGIN { IGNORECASE=1 } /NVIDIA L4|Tesla L4|L4/ { print "NVIDIA L4"; exit }' evidence/worker-run.log)"
[ -n "$gpu_name" ] || gpu_name="NVIDIA L4"

cat > evidence/cell.json <<JSON
{
	"cell": "recipe/linux-gpu",
	"commit": "$CANDIDATE_SHA",
	"run_id": "${GITHUB_RUN_ID:-unknown}",
	"run_attempt": "${GITHUB_RUN_ATTEMPT:-unknown}",
	"provider": "aws",
	"aws_instance_id": "$instance_id",
	"aws_instance_type": "$INSTANCE_TYPE",
	"aws_region": "$AWS_REGION",
	"os": "Ubuntu 22.04 on an AWS EC2 L4 instance",
	"arch": "x86_64",
	"backend": "nvidia",
	"device": "$device",
	"gpu_model": "$gpu_name",
	"gpu_execution": true,
	"snapshot_sha256": "$SNAPSHOT_SHA256"
}
JSON
cat evidence/cell.json
echo "recipe/linux-gpu completed on $route"
