#!/usr/bin/env bash
set -euo pipefail

export NVCF_BACKEND="${NVCF_CUSTOM_BACKEND:-$NVCF_BACKEND}"

# Determine HELM_IMAGE_TAG based on pipeline source
if [[ "$TEST_END_TO_END" == "True" ]]; then
  # this can be overridden by passing the HELM_IMAGE_TAG var in cicd variables
  export HELM_IMAGE_TAG=${HELM_IMAGE_TAG:-$(cosmos-curate nvcf image list-image-detail --iname "$NVCF_DEV_BASE_IMAGE" |grep latestTag |sed "s/['│,]//g" |awk '{print $2}')}
elif [[ "$CI_PIPELINE_SOURCE" == "web" ]]; then
  # Web triggers use staging images unless e2e is specified, then tests latest dev image
  export HELM_IMAGE_TAG=${HELM_IMAGE_TAG:-$(cosmos-curate nvcf image list-image-detail --iname "$NVCF_STAGING_BASE_IMAGE" |grep latestTag |sed "s/['│,]//g" |awk '{print $2}')}
else
  export HELM_IMAGE_TAG="${CI_COMMIT_TIMESTAMP%%T*}_${CI_COMMIT_SHORT_SHA}"
fi

# Determine NVCF_IMAGE based on pipeline source
if [[ "$CI_PIPELINE_SOURCE" == "web" && "$TEST_END_TO_END" != "True" ]]; then
  NVCF_IMAGE=${NVCF_STAGING_IMAGE}:$HELM_IMAGE_TAG
else
  NVCF_IMAGE=${NVCF_DEV_IMAGE}:$HELM_IMAGE_TAG
fi

# NOTE: create_helm_file.json will now contain keys/certificates so do not print its contents
envsubst < examples/nvcf/ci/ci_create_helm_default.json.template > create_helm_file.json
envsubst < examples/nvcf/ci/ci_deploy_helm_default.json.template > deploy_helm_file.json

# Update deploy_helm_file.json with the correct image repository
if [[ "$CI_PIPELINE_SOURCE" == "web" && "$TEST_END_TO_END" != "True" ]]; then
  jq --arg image_repository "$NVCF_STAGING_IMAGE" '.configuration.image.repository = $image_repository' < deploy_helm_file.json > deploy_helm_file.json.new
else
  jq --arg image_repository "$NVCF_DEV_IMAGE" '.configuration.image.repository = $image_repository' < deploy_helm_file.json > deploy_helm_file.json.new
fi
mv deploy_helm_file.json.new deploy_helm_file.json

# Echo deployment information
echo "NVCF DEPLOYMENT"
echo "*************************************"
echo "NVCF_IMAGE:         $NVCF_IMAGE"
echo "NVCF_BACKEND:       $NVCF_BACKEND"
echo "NVCF_GPU_TYPE:      $NVCF_GPU_TYPE"
echo "NVCF_INSTANCE_TYPE: $NVCF_INSTANCE_TYPE"
echo "AWS_DEFAULT_REGION: $AWS_DEFAULT_REGION"
echo "HELM_CHART_NAME:    $CICD_HELM_CHART_NAME"
echo "HELM_CHART_VERSION: $CICD_HELM_CHART_VERSION"
echo "*************************************"

# Create NVCF function
cosmos-curate nvcf function create-function --name "${HELM_FUNCTION_NAME}" --health-ep /api/local_raylet_healthz --health-port 52365 --helm-chart "https://helm.ngc.nvidia.com/${NGC_NVCF_ORG}/charts/${CICD_HELM_CHART_NAME}-${CICD_HELM_CHART_VERSION}.tgz" --data-file create_helm_file.json

# Copy funcid.json from ~/.config to working directory
cp ~/.config/cosmos_curate/funcid.json funcid_working.json

# Deploy NVCF function
cosmos-curate nvcf function deploy-function --data-file deploy_helm_file.json --max-concurrency "$NVCF_MAX_CONCURRENCY" --instance-count "$NVCF_INSTANCE_COUNT"

# Wait for deployment to be active
while true; do
  status=$(cosmos-curate nvcf function get-deployment-detail | grep "Status")
  if [[ $status =~ "DEPLOYING" ]]; then
    echo "Waiting for deployment to be active... (retrying in 10 seconds)"
    sleep 10
  elif [[ $status =~ "ACTIVE" ]]; then
    echo "Deployment is active"
    break
  else
    echo "Error: Deployment status '$status' is not recognized."
    exit 1
  fi
done

# Invoke NVCF function
# Create S3 credentials file
echo -n "$AWS_CONFIG_FILE_CONTENTS" | base64 -d > aws_credentials
sed -i '/^endpoint/d' aws_credentials

# Determine whether to raise on pynvc error
export DECODE_RAISE_ON_ERROR=true

# Split-annotate pipeline
# default config to process 1 video
jq --arg output_clip_path "$S3_OUTPUT_CLIP_PATH" --arg input_video_path "$S3_INPUT_VIDEO_PATH" --argjson limit 1 \
'.args.output_clip_path = $output_clip_path | .args.input_video_path = $input_video_path | .args.limit = $limit' \
< examples/nvcf/function/invoke_video_split_full.json > invoke_split1.json
cosmos-curate nvcf function invoke-function --data-file invoke_split1.json --s3-config-file aws_credentials

# use GPU decoding & FP8/TP2 for Qwen to process 1 more video
if [ "$HELM_GPU_REQUESTS" -ge 2 ]; then
  export _QWEN_TP_SIZE="2"
else
  export _QWEN_TP_SIZE="1"
fi

jq --arg output_clip_path "$S3_OUTPUT_CLIP_PATH" --arg input_video_path "$S3_INPUT_VIDEO_PATH" --argjson limit 1 --arg transnetv2_frame_decoder_mode pynvc --argjson transnetv2_frame_decode_raise_on_pynvc_error "$DECODE_RAISE_ON_ERROR" --argjson transcode_use_hwaccel true --argjson qwen_use_fp8_weights true --argjson qwen_num_gpus_per_worker "$_QWEN_TP_SIZE" \
'.args.output_clip_path = $output_clip_path | .args.input_video_path = $input_video_path | .args.limit = $limit | .args.transnetv2_frame_decoder_mode = $transnetv2_frame_decoder_mode | .args.transnetv2_frame_decode_raise_on_pynvc_error = $transnetv2_frame_decode_raise_on_pynvc_error | .args.transcode_use_hwaccel = $transcode_use_hwaccel | .args.qwen_use_fp8_weights = $qwen_use_fp8_weights | .args.qwen_num_gpus_per_worker = $qwen_num_gpus_per_worker ' \
< examples/nvcf/function/invoke_video_split_full.json > invoke_split2.json
cosmos-curate nvcf function invoke-function --data-file invoke_split2.json --s3-config-file aws_credentials

# Dedup pipeline
jq --arg output_path "$S3_OUTPUT_DEDUP_PATH" --arg input_embeddings_path "$S3_OUTPUT_CLIP_PATH" '.args.output_path = $output_path | .args.input_embeddings_path = $input_embeddings_path' < examples/nvcf/function/invoke_video_dedup.json > invoke_dedup.json
cosmos-curate nvcf function invoke-function --data-file invoke_dedup.json --s3-config-file aws_credentials

# Shard-dataset pipeline
jq --arg output_dataset_path "$S3_OUTPUT_DATASET_PATH" --arg input_clip_path "$S3_OUTPUT_CLIP_PATH" --arg input_semantic_dedup_path "$S3_OUTPUT_DEDUP_PATH" '.args.output_dataset_path = $output_dataset_path | .args.input_clip_path = $input_clip_path | .args.input_semantic_dedup_path = $input_semantic_dedup_path' < examples/nvcf/function/invoke_video_shard.json > invoke_shard.json
cosmos-curate nvcf function invoke-function --data-file invoke_shard.json --s3-config-file aws_credentials
