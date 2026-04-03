#!/usr/bin/env bash
set -euo pipefail

echo "Running nvcf split benchmark"

# Build curator image
PERF_IMAGE_TAG="${CI_COMMIT_TIMESTAMP%%T*}_${CI_COMMIT_SHORT_SHA}"
cosmos-curate image build \
  --curator-path . \
  --image-name "${CURATOR_IMAGE}" \
  --image-tag "${PERF_IMAGE_TAG}"

PERF_FULL_IMAGE="${CURATOR_IMAGE}:${PERF_IMAGE_TAG}"
echo "Built image ${PERF_FULL_IMAGE} from curator commit [${CI_COMMIT_SHA}]"
docker push "${PERF_FULL_IMAGE}"
echo "Pushed image ${PERF_FULL_IMAGE} to GitLab registry"

PERF_NVCF_IMAGE="${PERF_NVCF_IMAGE_REPOSITORY}:${PERF_IMAGE_TAG}"
docker buildx imagetools create -t "${PERF_NVCF_IMAGE}" "${PERF_FULL_IMAGE}"
echo "Copied to image ${PERF_NVCF_IMAGE} from curator commit [${CI_COMMIT_SHA}]"

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
    PERF_S3_OUTPUT_DIR="${PERF_S3_ROOT_DIR}/${date_str}_nodes_${num_nodes}_caption_${caption}_${splitting_algorithm}";
    echo "PERF_S3_OUTPUT_DIR: ${PERF_S3_OUTPUT_DIR}"
    micromamba run -n curator python benchmarks/split_pipeline/nvcf_split_benchmark.py \
      --num-nodes "${num_nodes}" \
      --caption "${caption}" \
      --splitting-algorithm "${splitting_algorithm}" \
      --captioning-algorithm "qwen" \
      --funcid "${PERF_NVCF_FUNC_ID}" \
      --version "${PERF_NVCF_FUNC_VERSION}" \
      --image-repository "${PERF_NVCF_IMAGE_REPOSITORY}" \
      --image-tag "${PERF_IMAGE_TAG}" \
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
