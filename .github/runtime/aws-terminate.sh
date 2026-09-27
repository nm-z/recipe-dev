#!/usr/bin/env bash
# Saves the instance's console output, terminates the instance, and removes the
# run's S3 objects. Runs after every attempt, including failed and cancelled ones.
set -u

: "${AWS_REGION:?AWS_REGION is required}"
: "${AWS_BUCKET:?AWS_BUCKET is required}"
mkdir -p evidence

if [ -f aws-instance-id ]; then
	instance_id="$(sed -n '1p' aws-instance-id)"
	if aws ec2 get-console-output --region "$AWS_REGION" --instance-id "$instance_id" --latest --query Output --output text > evidence/console.log 2>&1; then
		echo "saved the console output of $instance_id"
	else
		echo "could not read the console output of $instance_id" >&2
	fi
	if aws ec2 terminate-instances --region "$AWS_REGION" --instance-ids "$instance_id" --query 'TerminatingInstances[0].CurrentState.Name' --output text; then
		echo "terminated $instance_id"
	else
		echo "could not terminate $instance_id; it powers itself off at its lifetime limit" >&2
	fi
else
	echo "no AWS instance to terminate"
fi

if [ -f aws-s3-prefix ]; then
	s3_prefix="$(sed -n '1p' aws-s3-prefix)"
	aws s3 rm --only-show-errors --recursive "s3://$AWS_BUCKET/$s3_prefix/" || echo "could not remove s3://$AWS_BUCKET/$s3_prefix/" >&2
fi
exit 0
