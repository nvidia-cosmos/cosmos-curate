#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Running nvcf split benchmark"

# STAGING_IMAGE_NAME/STAGING_TAG from resolve job dotenv.
if [[ -z "${STAGING_IMAGE_NAME:-}" ]]; then
  echo "ERROR: STAGING_IMAGE_NAME is unset (needs resolve_nvcf_staging_tag dotenv)" >&2
  exit 1
fi
if [[ -z "${STAGING_TAG:-}" ]]; then
  echo "ERROR: STAGING_TAG is unset (needs resolve_nvcf_staging_tag dotenv or explicit setting)" >&2
  exit 1
fi

echo "Skopeo copy nvcr.io/${NGC_NVCF_ORG}/${STAGING_IMAGE_NAME}:${STAGING_TAG} -> nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${STAGING_IMAGE_NAME}:${STAGING_TAG}"
skopeo copy --all \
  --src-creds "\$oauthtoken:${NGC_REGISTRY_KEY}" \
  --dest-creds "\$oauthtoken:${PERF_REGISTRY_KEY}" \
  "docker://nvcr.io/${NGC_NVCF_ORG}/${STAGING_IMAGE_NAME}:${STAGING_TAG}" \
  "docker://nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${STAGING_IMAGE_NAME}:${STAGING_TAG}"
echo "Published nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${STAGING_IMAGE_NAME}:${STAGING_TAG}"

date_str=$(date +%Y%m%d%H%M%S)
LIMIT_INPUT_VIDEOS=5000
export RUST_BACKTRACE=1

# Run benchmark
# Defaults: caption=1 (performance-critical path), nodes=4 2, algorithms=transnetv2 fixed-stride.
# Override via env vars for targeted ad-hoc runs without editing the script.
caption="${CAPTION:-1}"
IFS=' ' read -ra _num_nodes_list      <<< "${NUM_NODES_LIST:-4 2}"
IFS=' ' read -ra _splitting_algo_list <<< "${SPLITTING_ALGORITHM_LIST:-transnetv2 fixed-stride}"
for num_nodes in "${_num_nodes_list[@]}"; do
  for splitting_algorithm in "${_splitting_algo_list[@]}"; do
    PERF_S3_OUTPUT_DIR="${PERF_S3_ROOT_DIR}/${date_str}_nodes_${num_nodes}_caption_${caption}_${splitting_algorithm}"
    echo "PERF_S3_OUTPUT_DIR: ${PERF_S3_OUTPUT_DIR}"
    micromamba run -n curator python benchmarks/split_pipeline/nvcf_split_benchmark.py \
      --num-nodes "${num_nodes}" \
      --caption "${caption}" \
      --splitting-algorithm "${splitting_algorithm}" \
      --captioning-algorithm "qwen" \
      --funcid "${PERF_NVCF_FUNC_ID}" \
      --version "${PERF_NVCF_FUNC_VERSION}" \
      --image-repository "nvcr.io/${PERF_NGC_NVCF_ORG_ID}/${STAGING_IMAGE_NAME}" \
      --image-tag "${STAGING_TAG}" \
      --metrics-endpoint "${PERF_NVCF_METRICS_ENDPOINT}" \
      --backend "${PERF_NVCF_BACKEND}" \
      --gpu "${PERF_NVCF_GPU}" \
      --instance-type "${PERF_NVCF_INSTANCE_TYPE}" \
      --s3-input-prefix "${PERF_S3_INPUT_DIR}" \
      --s3-output-prefix "${PERF_S3_OUTPUT_DIR}" \
      --gpus-per-node 8 \
      --max-concurrency 2 \
      --kratos-metrics-endpoint "${PERF_KRATOS_METRICS_ENDPOINT}" \
      --kratos-bearer-url "${PERF_KRATOS_BEARER_URL}" \
      --limit "${LIMIT_INPUT_VIDEOS}" \
      --report-metrics-to-kratos
  done
done
