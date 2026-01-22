#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

# Confirm we're only using the credentials from file
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY

# Test s3 access first to ensure credentials are valid
if ! out=$(aws s3 ls "${S3_INPUT_VIDEO_PATH}" 2>&1); then
  echo "Error: Failed to access S3: $out"
  sha256sum aws_credentials
  exit 1
fi

# Wait for split-annotate summary file
if ! wait_for_s3_file "${S3_FILE_SPLIT_SUMMARY}"; then
    echo "S3 split summary not found, retry limit exceeded."
    exit 1
fi

# Validate split-annotate summary content
echo "Validating $S3_FILE_SPLIT_SUMMARY ..."
if ! JSON_CONTENT=$(validate_s3_json "$S3_FILE_SPLIT_SUMMARY"); then
    exit 1
fi

num_videos=$(jq -r ".num_processed_videos" <<< "$JSON_CONTENT")
if [ "$num_videos" -ne 2 ]; then
    echo "Error: There should be 2 videos processed, but found $num_videos"
    exit 1
fi
num_clips=$(jq -r ".num_clips_transcoded" <<< "$JSON_CONTENT")
num_clips_with_caption=$(jq -r ".num_clips_with_caption" <<< "$JSON_CONTENT")
if [ "$num_clips" -lt 2 ]; then
    echo "Error: There should be at least 2 clips transcoded, but found $num_clips"
    exit 1
fi
if [ "$num_clips_with_caption" -ne "$num_clips" ]; then
    echo "Error: All clips should have captions, but found $num_clips_with_caption out of $num_clips"
    exit 1
fi
echo "Split-annotate pipeline finished with $num_videos processed and $num_clips_with_caption captioned"
echo "Split-annotate pipeline validation successful"

# Wait for dedup summary file
if ! wait_for_s3_file "${S3_FILE_DEDUP_SUMMARY}"; then
    echo "S3 dedup summary not found, retry limit exceeded."
    exit 1
fi
echo "Dedup pipeline validation successful"

# Wait for shard summary file
if ! wait_for_s3_file "${S3_FILE_SHARD_SUMMARY}"; then
    echo "S3 shard summary not found, retry limit exceeded."
    exit 1
fi
echo "Shard-dataset pipeline validation successful"
