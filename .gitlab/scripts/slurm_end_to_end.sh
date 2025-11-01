#!/usr/bin/env bash
# Run end-to-end pipelines on SLURM cluster

set -euo pipefail

mkdir -p "${ENROOT_CONFIG_PATH}"
echo "machine ${CI_REGISTRY/:5005/} login gitlab-ci-token password ${CI_JOB_TOKEN}" > "${ENROOT_CONFIG_PATH}/.credentials"

IMAGE_TAG="${CI_COMMIT_TIMESTAMP%%T*}_${CI_COMMIT_SHORT_SHA}"
FULL_IMAGE=${CURATOR_IMAGE}:${IMAGE_TAG}
BUILD_IMAGE_NAME_SBATCH="${FULL_IMAGE/:5005\///}"

DATA_DIR=/lustre/fsw/coreai_dlalgo_ci/datasets/nemo_curator/video
MODEL_DIR=/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/models
AWS_CREDS_PATH=/lustre/fsw/coreai_dlalgo_ci/datasets/nemo_curator/video/awscreds

if [[ ! -e "${AWS_CREDS_PATH}" ]]; then
  echo "AWS credentials file not found at ${AWS_CREDS_PATH}" >&2
  exit 1
fi

MOUNTS=(
  "${DATA_DIR}:/config/data"
  "${MODEL_DIR}:/config/models"
  "${AWS_CREDS_PATH}:/creds/s3_creds"
  "${CI_PROJECT_DIR}:/config/project"
)
MOUNTS_STR=$(IFS=, ; echo "${MOUNTS[*]}")

LOG_DIR="${CI_PROJECT_DIR}/slurm_logs"
REMOTE_FILES_DIR="${CI_PROJECT_DIR}/slurm_remote_files"
mkdir -p "${LOG_DIR}" "${REMOTE_FILES_DIR}"

export ENROOT_CONFIG_PATH
export SLURM_LOG_DIR="${LOG_DIR}"

SLURM_E2E_OUTPUT_PREFIX="${S3_OUTPUT_PATH}/cosmos-curator-slurm"
SLURM_E2E_OUTPUT_CLIP_PATH="${SLURM_E2E_OUTPUT_PREFIX}/raw_clips"
SLURM_E2E_OUTPUT_DEDUP_PATH="${SLURM_E2E_OUTPUT_PREFIX}/dedup_results"
SLURM_E2E_OUTPUT_DATASET_PATH="${SLURM_E2E_OUTPUT_PREFIX}/datasets"

CONTAINER_ENV=(
  "AWS_SHARED_CREDENTIALS_FILE=/creds/s3_creds"
  "AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}"
  "MODEL_WEIGHTS_PREFIX=/config/models"
  "S3_INPUT_VIDEO_PATH=${S3_INPUT_VIDEO_PATH}"
  "SLURM_E2E_S3_PROFILE_NAME=${SLURM_E2E_S3_PROFILE_NAME}"
  "SLURM_E2E_OUTPUT_CLIP_PATH=${SLURM_E2E_OUTPUT_CLIP_PATH}"
  "SLURM_E2E_OUTPUT_DEDUP_PATH=${SLURM_E2E_OUTPUT_DEDUP_PATH}"
  "SLURM_E2E_OUTPUT_DATASET_PATH=${SLURM_E2E_OUTPUT_DATASET_PATH}"
)
CONTAINER_ENV_STR=$(IFS=, ; echo "${CONTAINER_ENV[*]}")

LOGIN_NODE="${SLURM_LOGIN_NODE:-$(hostname -f)}"
JOB_NAME="${SLURM_ACCOUNT}-cosmos_curate_e2e.${CI_JOB_ID}"

submit_cmd=(
  cosmos-curate slurm submit
  --login-node "${LOGIN_NODE}"
  --account "${SLURM_ACCOUNT}"
  --partition "${SLURM_PARTITION}"
  --remote-files-path "${REMOTE_FILES_DIR}"
  --container-image "${BUILD_IMAGE_NAME_SBATCH}"
  --container-mounts "${MOUNTS_STR}"
  --environment "${CONTAINER_ENV_STR}"
  --job-name "${JOB_NAME}"
  --log-dir "${LOG_DIR}"
  --time "02:00:00"
)
if [[ -n "${SLURM_GRES:-}" ]]; then
  submit_cmd+=(--gres "${SLURM_GRES}")
fi
submit_cmd+=(-- pixi run bash /config/project/examples/slurm/ci_run_end_to_end.sh)
"${submit_cmd[@]}" | tee slurm_submit.log

JOB_ID=$(awk '/Job submitted with ID:/{print $NF}' slurm_submit.log | tail -n 1)
if [[ -z "${JOB_ID}" ]]; then
  echo "Could not determine SLURM job ID from submission output" >&2
  cat slurm_submit.log
  exit 1
fi

echo "Submitted SLURM end-to-end job ${JOB_ID}"

wait_for_job() {
  local job_id=$1
  local max_attempts=120  # Align with the 02:00:00 sbatch time limit (120 minutes)
  local attempt=0
  while (( attempt < max_attempts )); do
    if squeue -h -j "${job_id}" >/dev/null 2>&1; then
      echo "[$(date -Ins)] Job ${job_id} still running..."
    else
      local state
      state=$(sacct -j "${job_id}" -o State -n | head -n 1 | tr -d ' ')
      echo "[$(date -Ins)] Job ${job_id} completed with state ${state}"
      if [[ "${state}" == COMPLETED* ]]; then
        return 0
      fi
      return 1
    fi
    sleep 60
    attempt=$((attempt + 1))
  done
  echo "Timeout waiting for job ${job_id}" >&2
  return 1
}

if ! wait_for_job "${JOB_ID}"; then
  LOG_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.log"
  if [[ -f "${LOG_FILE}" ]]; then
    echo "---- SLURM job log (${LOG_FILE}) ----"
    tail -n 200 "${LOG_FILE}"
  else
    echo "SLURM log file ${LOG_FILE} was not found" >&2
  fi
  exit 1
fi

LOG_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.log"
if [[ -f "${LOG_FILE}" ]]; then
  echo "Collected SLURM log at ${LOG_FILE}"
fi

export AWS_SHARED_CREDENTIALS_FILE="${AWS_CREDS_PATH}"

json_is_valid() {
  python -c 'import json, sys; json.load(sys.stdin)' >/dev/null 2>&1
}

validate_path() {
  local target=$1
  local retries=10
  local attempt=0
  while (( attempt < retries )); do
    if aws s3 ls "${target}" >/dev/null 2>&1; then
      echo "Validated S3 object ${target}"
      return 0
    fi
    echo "Waiting for ${target} to appear in S3 (${attempt}/${retries})..."
    sleep 5
    attempt=$((attempt + 1))
  done
  echo "S3 object ${target} not found after retries" >&2
  return 1
}

validate_path "${SLURM_E2E_OUTPUT_CLIP_PATH}/summary.json"
validate_path "${SLURM_E2E_OUTPUT_DEDUP_PATH}/extraction/dedup_summary_0.01.csv"
validate_path "${SLURM_E2E_OUTPUT_DATASET_PATH}/v0/wdinfo_list.csv"

SUMMARY_CONTENT=$(aws s3 cp "${SLURM_E2E_OUTPUT_CLIP_PATH}/summary.json" -)
if ! json_is_valid <<<"${SUMMARY_CONTENT}"; then
  echo "Split summary JSON is invalid" >&2
  exit 1
fi
