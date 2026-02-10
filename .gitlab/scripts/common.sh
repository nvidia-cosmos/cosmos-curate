#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Common functions for CI scripts
# Usage: source "$(dirname "$0")/common.sh"

# Decode base64 S3 credentials and export path
# Args: output_path (default: /tmp/s3_creds_file)
setup_s3_credentials() {
    local output_path="${1:-/tmp/s3_creds_file}"
    echo -n "$AWS_CONFIG_FILE_CONTENTS" | base64 -d > "$output_path"
    export COSMOS_S3_PROFILE_PATH="$output_path"
    echo "S3 credentials written to $output_path"
}

# Generate consistent image tag from CI variables
# Mirrors the compute_image_tag anchor in .gitlab-ci.yml
get_image_tag() {
    local branch_prefix
    if [ -n "${CI_MERGE_REQUEST_TARGET_BRANCH_NAME:-}" ]; then
        branch_prefix="${CI_MERGE_REQUEST_TARGET_BRANCH_NAME##*/}"
    else
        branch_prefix="${CI_COMMIT_BRANCH##*/}"
    fi
    echo "${branch_prefix}_${CI_COMMIT_TIMESTAMP%%T*}_${CI_COMMIT_SHORT_SHA}"
}

# Canonical NGC credentials
# NGC_NVCF_ORG is set in .gitlab-ci.yml (defaults to PERF org)
# NGC_API_KEY defaults to PERF key (vault secrets not available at CI variable definition time)
: "${NGC_API_KEY:=${PERF_NGC_NVCF_API_KEY:-}}"
: "${NGC_ORG:=${NGC_NVCF_ORG:-${PERF_NGC_NVCF_ORG_ID:-}}}"

# Configure environment for NGC model downloads
setup_ngc_model_download() {
    export NVCF_MULTI_NODE=true
    export NGC_NVCF_API_KEY="${NGC_API_KEY}"
    export NGC_NVCF_ORG="${NGC_ORG}"
    echo "NGC model download configured (org: ${NGC_ORG})"
}

# Wait for S3 file to appear
# Args: s3_path, max_attempts (default: 10), sleep_seconds (default: 5)
wait_for_s3_file() {
    local path="$1"
    local max="${2:-10}"
    local sleep_sec="${3:-5}"

    for ((i=0; i<max; i++)); do
        if aws s3 ls "$path" &>/dev/null; then
            echo "Found: $path"
            return 0
        fi
        echo "Waiting for $path... ($((i+1))/$max)"
        sleep "$sleep_sec"
    done
    echo "ERROR: $path not found after $max attempts"
    return 1
}

# Validate JSON from S3
# Args: s3_path
validate_s3_json() {
    local path="$1"
    local content
    if ! content=$(aws s3 cp "$path" - 2>/dev/null); then
        echo "ERROR: Failed to read $path"
        return 1
    fi
    if ! jq empty <<< "$content" 2>/dev/null; then
        echo "ERROR: Invalid JSON in $path"
        return 1
    fi
    echo "$content"
}

# Standard pipeline run arguments for reduced CPU usage (fits 8-core nodes)
# Returns args string suitable for appending to pipeline command
get_reduced_cpu_pipeline_args() {
    echo "--transnetv2-frame-decode-cpus-per-worker 1 --transcode-cpus-per-worker 1 --clip-extraction-cpus-per-worker 1"
}
