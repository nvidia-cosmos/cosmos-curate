#!/usr/bin/env bash
set -euo pipefail

# Wait for split-annotate summary file
set +e  # Disable exit on error
# Confirm we're only using the credentials from file
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
# Test s3 access first to ensure credentials are valid
out=$(aws s3 ls "${S3_INPUT_VIDEO_PATH}" 2>&1); rc=$?
if [ $rc -ne 0 ]; then
  echo "Error: Failed to access S3: $out"
  sha256sum aws_credentials
  exit 1
fi
i=0
max=10
file_found=false
while [ $i -lt $max ]; do
    if aws s3 ls "${S3_FILE_SPLIT_SUMMARY}" &> /dev/null; then
        echo "Found S3 split-annotate summary file: ${S3_FILE_SPLIT_SUMMARY}"
        file_found=true
        break
    else
        echo "Waiting for S3 data in ${S3_FILE_SPLIT_SUMMARY} ... ($i of $max, retrying in 5 seconds)"
        sleep 5
        ((i++))
    fi
done
set -e
if [ "$file_found" = false ]; then
    echo "S3 command failed, retry limit exceeded."
    exit 1
fi

# Validate split-annotate summary content
echo "Validating $S3_FILE_SPLIT_SUMMARY ..."
if ! JSON_CONTENT=$(aws s3 cp "$S3_FILE_SPLIT_SUMMARY" - 2>/dev/null); then
    echo "Error reading from S3: $S3_FILE_SPLIT_SUMMARY"
    exit 1
fi
if ! jq empty <<< "$JSON_CONTENT" 2>/dev/null; then
    echo "Error: Invalid JSON structure"
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
set +e  # Disable exit on error
i=0
max=10
file_found=false
while [ $i -lt $max ]; do
    if aws s3 ls "${S3_FILE_DEDUP_SUMMARY}" &> /dev/null; then
        echo "Found S3 dedup summary file: ${S3_FILE_DEDUP_SUMMARY}"
        file_found=true
        break
    else
        echo "Waiting for S3 data in ${S3_FILE_DEDUP_SUMMARY} ... ($i of $max, retrying in 5 seconds)"
        sleep 5
        ((i++))
    fi
done
set -e
if [ "$file_found" = false ]; then
    echo "S3 command failed, retry limit exceeded."
    exit 1
fi
echo "Dedup pipeline validation successful"

# Wait for shard summary file
set +e  # Disable exit on error
i=0
max=10
file_found=false
while [ $i -lt $max ]; do
    if aws s3 ls "${S3_FILE_SHARD_SUMMARY}" &> /dev/null; then
        echo "Found S3 shard summary file: ${S3_FILE_SHARD_SUMMARY}"
        file_found=true
        break
    else
        echo "Waiting for S3 data in ${S3_FILE_SHARD_SUMMARY} ... ($i of $max, retrying in 5 seconds)"
        sleep 5
        ((i++))
    fi
done
set -e
if [ "$file_found" = false ]; then
    echo "S3 command failed, retry limit exceeded."
    exit 1
fi
echo "Shard-dataset pipeline validation successful"
