#!/usr/bin/env bash
# Runs the immutable candidate archive on one Arch Linux NVIDIA EC2 worker.
#
# The AWS credential stays in this controller process. The worker is reached
# over SSH with a key made for this run, through a security group that admits
# only this runner's address, and receives only the archive, the trusted
# runtime and the guest script. aws-cleanup.sh terminates everything the run
# created, tagged recipe-worker=$WORKER.
set -euo pipefail
: "${AWS_ACCESS_KEY_ID:?the AWS credential is required}"
: "${AWS_SECRET_ACCESS_KEY:?the AWS credential is required}"
: "${CANDIDATE_SHA:?CANDIDATE_SHA is required}"
: "${SNAPSHOT:?SNAPSHOT is required}"
: "${SNAPSHOT_SHA256:?SNAPSHOT_SHA256 is required}"
: "${TRUSTED_RUNTIME:?TRUSTED_RUNTIME is required}"

export AWS_DEFAULT_REGION="${AWS_REGION:-us-east-1}"
export AWS_PAGER=""
INSTANCE_TYPE="${AWS_NV_INSTANCE_TYPE:-g4dn.xlarge}"
# The Arch Linux project's published images. The owner is a repository
# variable so a different publisher can be chosen without a code change.
ARCH_AMI_OWNER="${AWS_ARCH_AMI_OWNER:-647457786197}"
ARCH_AMI_NAME="${AWS_ARCH_AMI_NAME:-arch-linux-*}"
SSH_USER="${AWS_ARCH_SSH_USER:-arch}"
RUN_ID="${GITHUB_RUN_ID:-manual}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
WORKER="recipe-lnv-${RUN_ID}-${RUN_ATTEMPT}"
KEY="${RUNNER_TEMP:-/tmp}/$WORKER.key"
mkdir -p evidence
printf '%s\n' "$WORKER" > aws-worker-name

blocker() {
	local name="$1"
	local detail="$2"
	local resolution="$3"
	jq -n --arg blocker "$name" --arg detail "$detail" --arg resolution "$resolution" \
		'{blocker: $blocker, detail: $detail, resolution: $resolution}' > evidence/blocker.json
	cat evidence/blocker.json
	exit 1
}

echo "== checking the AWS login =="
if ! aws sts get-caller-identity --query Arn --output text > evidence/aws-identity.txt 2> evidence/aws-identity.log; then
	blocker "aws-credential-rejected" "$(cat evidence/aws-identity.log)" \
		"AWS refused AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY from the recipe-windows-gpu environment; store a current access key there."
fi
echo "signed in as $(cat evidence/aws-identity.txt) in $AWS_DEFAULT_REGION"

echo "== resolving the Arch Linux image =="
ami="$(aws ec2 describe-images --owners "$ARCH_AMI_OWNER" \
	--filters "Name=name,Values=$ARCH_AMI_NAME" Name=architecture,Values=x86_64 Name=state,Values=available \
	--query 'sort_by(Images, &CreationDate)[-1].[ImageId,RootDeviceName,Name]' --output text 2> evidence/aws-image.log || true)"
if [ -z "$ami" ] || [ "${ami%%[[:space:]]*}" = None ]; then
	candidates="$(aws ec2 describe-images --filters "Name=name,Values=arch-linux*" Name=architecture,Values=x86_64 \
		--query 'sort_by(Images, &CreationDate)[-8:].[OwnerId,ImageId,Name]' --output text 2>&1 || true)"
	blocker "aws-arch-image-unresolved" "No $ARCH_AMI_NAME image from owner $ARCH_AMI_OWNER in $AWS_DEFAULT_REGION. Newest public arch-linux images: $candidates" \
		"Set the AWS_ARCH_AMI_OWNER repository variable to the Arch Linux publisher's account."
fi
read -r ami_id root_device _ <<< "$ami"
echo "image $ami"

echo "== checking the G instance quota =="
vcpus="$(aws ec2 describe-instance-types --instance-types "$INSTANCE_TYPE" --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' --output text)"
# L-DB2E81BA is "Running On-Demand G and VT instances", counted in vCPUs. An
# approved increase can take a while to apply, so a short quota is waited on
# for AWS_QUOTA_WAIT_SECONDS before it counts as the blocker.
quota_deadline=$(( $(date +%s) + ${AWS_QUOTA_WAIT_SECONDS:-1200} ))
while quota="$(aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value' --output text 2> evidence/aws-quota.log)"; do
	echo "quota=$quota vCPUs, $INSTANCE_TYPE needs $vcpus"
	awk -v quota="$quota" -v need="$vcpus" 'BEGIN { exit !(quota < need) }' || break
	if [ "$(date +%s)" -ge "$quota_deadline" ]; then
		blocker "aws-gpu-quota" "Running On-Demand G and VT instances is $quota vCPUs in $AWS_DEFAULT_REGION; $INSTANCE_TYPE needs $vcpus." \
			"Request at least $vcpus vCPUs for that quota in $AWS_DEFAULT_REGION (Service Quotas, EC2), and wait until the applied value shows it."
	fi
	sleep 60
done
[ -n "${quota:-}" ] || echo "the quota is unreadable with this credential; the launch will report it"

echo "== creating the per-run key and security group =="
rm -f "$KEY" "$KEY.pub"
ssh-keygen -q -t ed25519 -N '' -C "$WORKER" -f "$KEY"
aws ec2 import-key-pair --key-name "$WORKER" --public-key-material "fileb://$KEY.pub" \
	--tag-specifications "ResourceType=key-pair,Tags=[{Key=recipe-worker,Value=$WORKER}]" > /dev/null
vpc="$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)"
if [ -z "$vpc" ] || [ "$vpc" = None ]; then
	blocker "aws-default-vpc-absent" "$AWS_DEFAULT_REGION has no default VPC." "Create a default VPC in $AWS_DEFAULT_REGION."
fi
group="$(aws ec2 create-security-group --group-name "$WORKER" --description "Recipe CI worker $WORKER" --vpc-id "$vpc" \
	--tag-specifications "ResourceType=security-group,Tags=[{Key=recipe-worker,Value=$WORKER}]" --query GroupId --output text)"
runner_ip="$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')"
aws ec2 authorize-security-group-ingress --group-id "$group" --protocol tcp --port 22 --cidr "$runner_ip/32" > /dev/null
echo "security group $group admits SSH from $runner_ip only"

echo "== launching $INSTANCE_TYPE =="
if ! aws ec2 run-instances \
	--image-id "$ami_id" \
	--instance-type "$INSTANCE_TYPE" \
	--key-name "$WORKER" \
	--security-group-ids "$group" \
	--associate-public-ip-address \
	--instance-initiated-shutdown-behavior terminate \
	--block-device-mappings "DeviceName=$root_device,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}" \
	--tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$WORKER},{Key=recipe-worker,Value=$WORKER}]" "ResourceType=volume,Tags=[{Key=recipe-worker,Value=$WORKER}]" \
	--query 'Instances[0].InstanceId' --output text > aws-instance-id 2> evidence/aws-launch.log; then
	blocker "aws-gpu-provisioning-failed" "$(cat evidence/aws-launch.log)" \
		"EC2 refused $INSTANCE_TYPE with $ami_id in $AWS_DEFAULT_REGION; the detail names the capacity, quota or permission cause."
fi
instance="$(cat aws-instance-id)"
echo "instance $instance"
aws ec2 wait instance-running --instance-ids "$instance"
address="$(aws ec2 describe-instances --instance-ids "$instance" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)"
echo "running at $address"

SSH_OPTIONS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=10 -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
# The command is composed here on purpose; the worker receives it as one string.
# shellcheck disable=SC2029
guest() {
	ssh "${SSH_OPTIONS[@]}" "$SSH_USER@$address" "$@"
}
wait_for_ssh() {
	local deadline=$(( $(date +%s) + 900 ))
	until guest true 2> /dev/null; do
		if [ "$(date +%s)" -ge "$deadline" ]; then
			aws ec2 get-console-output --instance-id "$instance" --latest --output text > evidence/aws-console.log 2>&1 || true
			blocker "aws-worker-unreachable" "SSH to $SSH_USER@$address did not answer within 15 minutes; evidence/aws-console.log holds the console." \
				"Check the image's login user (AWS_ARCH_SSH_USER) and that it boots on $INSTANCE_TYPE."
		fi
		sleep 10
	done
}
wait_for_ssh
echo "SSH is up as $SSH_USER"

echo "== transferring the immutable snapshot =="
[ "$(sha256sum "$SNAPSHOT" | cut -d' ' -f1)" = "$SNAPSHOT_SHA256" ] || { echo "snapshot checksum mismatch before upload" >&2; exit 1; }
[ -f "$TRUSTED_RUNTIME/suite.rs" ] || { echo "trusted suite is absent" >&2; exit 1; }
[ -d "$TRUSTED_RUNTIME/data" ] || { echo "trusted suite data is absent" >&2; exit 1; }
tar -czf trusted-runtime.tar.gz -C "$TRUSTED_RUNTIME" suite.rs data
runtime_sha256="$(sha256sum trusted-runtime.tar.gz | cut -d' ' -f1)"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
guest "sudo mkdir -p /var/lib/recipe/incoming && sudo chown $SSH_USER /var/lib/recipe/incoming"
scp -q "${SSH_OPTIONS[@]}" "$SNAPSHOT" "$SSH_USER@$address:/var/lib/recipe/incoming/snapshot.tar.gz"
scp -q "${SSH_OPTIONS[@]}" trusted-runtime.tar.gz "$SSH_USER@$address:/var/lib/recipe/incoming/trusted-runtime.tar.gz"
scp -q "${SSH_OPTIONS[@]}" "$script_dir/aws-guest.sh" "$SSH_USER@$address:/var/lib/recipe/incoming/aws-guest.sh"

run_phase() {
	local phase="$1"
	local deadline="$2"
	local marker="$3"
	local status
	echo "== guest phase: $phase =="
	set +e
	timeout --signal=TERM --kill-after=30s "${deadline}s" \
		ssh "${SSH_OPTIONS[@]}" "$SSH_USER@$address" \
		sudo bash /var/lib/recipe/incoming/aws-guest.sh "phase=$phase" "candidateSha=$CANDIDATE_SHA" "snapshotSha256=$SNAPSHOT_SHA256" "runtimeSuiteSha256=$runtime_sha256" \
		2>&1 | tee "evidence/$phase.log"
	status="${PIPESTATUS[0]}"
	set -e
	echo "$phase returned with status $status"
	[ "$status" -ne 124 ] || { echo "$phase exceeded its ${deadline}s deadline" >&2; return 1; }
	grep -Fq "$marker" "evidence/$phase.log" || { echo "$phase did not report '$marker'" >&2; return 1; }
}

run_phase driver 2400 "DRIVER EXIT 0"
echo "== rebooting into the upgraded kernel and the NVIDIA module =="
boot_before="$(guest cat /proc/sys/kernel/random/boot_id)"
guest "sudo systemctl reboot" || true
reboot_deadline=$(( $(date +%s) + 900 ))
while :; do
	sleep 15
	wait_for_ssh
	boot_after="$(guest cat /proc/sys/kernel/random/boot_id 2> /dev/null || true)"
	[ -z "$boot_after" ] || [ "$boot_after" = "$boot_before" ] || break
	[ "$(date +%s)" -lt "$reboot_deadline" ] || blocker "aws-worker-reboot-failed" "The worker still reports boot $boot_before 15 minutes after the reboot." "Check the console output of the worker."
done
echo "rebooted: boot $boot_after"
run_phase execute 2400 "GUEST EXIT 0"

scp -q "${SSH_OPTIONS[@]}" "$SSH_USER@$address:/var/lib/recipe/suite.json" evidence/suite.json
scp -q "${SSH_OPTIONS[@]}" "$SSH_USER@$address:/var/lib/recipe/run.log" evidence/worker-run.log
jq -e '.executed == 8 and .failed == 0' evidence/suite.json > /dev/null || { echo "the suite evidence does not contain eight passing checks" >&2; exit 1; }
route="$(sed -n 's/^executed on //p' evidence/execute.log | tail -n 1 | tr -d '\r')"
device="${route##*:}"
case "$device" in nv*) ;; *) echo "the evidence does not show an NVIDIA device: '$route'" >&2; exit 1 ;; esac
gpu_model="$(awk -F', ' '/^(Tesla|NVIDIA)/ { print $1; exit }' evidence/execute.log)"
cat > evidence/cell.json <<JSON
{
  "cell": "recipe/linux-gpu",
  "commit": "$CANDIDATE_SHA",
  "run_id": "$RUN_ID",
  "run_attempt": "$RUN_ATTEMPT",
  "provider": "aws",
  "aws_instance": "$instance",
  "aws_instance_type": "$INSTANCE_TYPE",
  "aws_image": "$ami",
  "os": "Arch Linux",
  "arch": "x86_64",
  "backend": "nvidia",
  "device": "$device",
  "gpu_model": "${gpu_model:-unknown}",
  "gpu_execution": true,
  "snapshot_sha256": "$SNAPSHOT_SHA256"
}
JSON
cat evidence/cell.json
echo "recipe/linux-gpu completed on $route"
