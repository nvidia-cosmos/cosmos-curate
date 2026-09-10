#!/usr/bin/env bash
# Run redistributable media smoke tests on the SLURM cluster.

set -euo pipefail

# shellcheck source=common.sh
source "$(dirname "$0")/common.sh"

DISTRIBUTABLE_SMOKE_MODE="${DISTRIBUTABLE_SMOKE_MODE:-bare}"
DISTRIBUTABLE_FFMPEG_PREFIX="${DISTRIBUTABLE_FFMPEG_PREFIX:-/lustre/fsw/coreai_dlalgo_ci/nemo_video_curator/ffmpeg/.pixi/envs/default}"

case "${DISTRIBUTABLE_SMOKE_MODE}" in
  bare|mounted-ffmpeg)
    ;;
  *)
    echo "Unsupported DISTRIBUTABLE_SMOKE_MODE=${DISTRIBUTABLE_SMOKE_MODE}" >&2
    exit 1
    ;;
esac

mkdir -p "${ENROOT_CONFIG_PATH}"
echo "machine ${CI_REGISTRY/:5005/} login gitlab-ci-token password ${CI_JOB_TOKEN}" > "${ENROOT_CONFIG_PATH}/.credentials"

IMAGE_TAG="$(get_image_tag)"
FULL_IMAGE="${CURATOR_DISTRIBUTABLE_IMAGE}:${IMAGE_TAG}"
BUILD_IMAGE_NAME_SBATCH="${FULL_IMAGE/:5005\///}"

LOG_DIR="${CI_PROJECT_DIR}/slurm_distributable_smoke_logs"
REMOTE_FILES_DIR="${CI_PROJECT_DIR}/slurm_distributable_smoke_remote_files"
mkdir -p "${LOG_DIR}" "${REMOTE_FILES_DIR}"

export ENROOT_CONFIG_PATH
export SLURM_LOG_DIR="${LOG_DIR}"

MOUNTS=(
  "${CI_PROJECT_DIR}:/config/project:ro"
)
if [[ "${DISTRIBUTABLE_SMOKE_MODE}" == "mounted-ffmpeg" ]]; then
  if [[ ! -d "${DISTRIBUTABLE_FFMPEG_PREFIX}" ]]; then
    echo "DISTRIBUTABLE_FFMPEG_PREFIX does not exist: ${DISTRIBUTABLE_FFMPEG_PREFIX}" >&2
    exit 1
  fi
  MOUNTS+=("${DISTRIBUTABLE_FFMPEG_PREFIX}:/opt/ffmpeg:ro")
fi
MOUNTS_STR=$(IFS=, ; echo "${MOUNTS[*]}")

CONTAINER_ENV=(
  "PYTHONUNBUFFERED=1"
  "CONDA_OVERRIDE_CUDA=13.0.3"
)
CONTAINER_ENV_STR=$(IFS=, ; echo "${CONTAINER_ENV[*]}")

LOGIN_NODE="${SLURM_LOGIN_NODE:-$(hostname -f)}"
JOB_MODE="${DISTRIBUTABLE_SMOKE_MODE//-/_}"
JOB_NAME="${SLURM_ACCOUNT}-cosmos_curator_distributable_${JOB_MODE}.${CI_JOB_ID}"

submit_cmd=(
  cosmos-curator slurm submit
  --login-node "${LOGIN_NODE}"
  --account "${SLURM_ACCOUNT}"
  --partition "${SLURM_PARTITION}"
  --remote-files-path "${REMOTE_FILES_DIR}"
  --container-image "${BUILD_IMAGE_NAME_SBATCH}"
  --container-mounts "${MOUNTS_STR}"
  --environment "${CONTAINER_ENV_STR}"
  --job-name "${JOB_NAME}"
  --log-dir "${LOG_DIR}"
  --time "00:15:00"
)
if [[ -n "${SLURM_GRES:-}" ]]; then
  submit_cmd+=(--gres "${SLURM_GRES}")
fi
submit_cmd+=(
  --
  pixi run --as-is -e default python /config/project/.gitlab/scripts/distributable_media_smoke.py
  --mode "${DISTRIBUTABLE_SMOKE_MODE}"
)
"${submit_cmd[@]}" | tee "slurm_distributable_smoke_${JOB_MODE}.submit.log"

JOB_ID=$(awk '/Job submitted with ID:/{print $NF}' "slurm_distributable_smoke_${JOB_MODE}.submit.log" | tail -n 1)
if [[ -z "${JOB_ID}" ]]; then
  echo "Could not determine SLURM job ID from submission output" >&2
  cat "slurm_distributable_smoke_${JOB_MODE}.submit.log"
  exit 1
fi

echo "Submitted redistributable media smoke job ${JOB_ID} (${DISTRIBUTABLE_SMOKE_MODE})"

LOG_FILE="${LOG_DIR}/${JOB_NAME}_${JOB_ID}.log"
echo "SLURM job log will stream from ${LOG_FILE} once it is created"

if ! monitor_slurm_job "${JOB_ID}" "${LOG_FILE}" 30 30; then
  exit 1
fi
