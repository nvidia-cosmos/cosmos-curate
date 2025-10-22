#!/bin/bash

# check if env var SLURM_USER_DIR is defined
if [ -z "$SLURM_USER_DIR" ]; then
    echo "Error: SLURM_USER_DIR is not defined. Please set it to your SLURM user directory."
    return
fi

echo "Using $SLURM_USER_DIR as user directory on Slurm cluster."
echo "--------------------------------"

export SLURM_LOG_DIR="${SLURM_USER_DIR}/job_logs"
export SLURM_COSMOS_CURATE_CONFIG_DIR="${SLURM_USER_DIR}/.config/cosmos_curate"
export SLURM_AWS_CREDS_DIR="${SLURM_USER_DIR}/.aws"
export SLURM_AZURE_CREDS_DIR="${SLURM_USER_DIR}/.azure"
export SLURM_WORKSPACE="${SLURM_USER_DIR}/cosmos_curate_local_workspace"
export SLURM_IMAGE_DIR="${SLURM_USER_DIR}/container_images"
export SLURM_SOURCE_DIR="${SLURM_USER_DIR}/src/cosmos-curate"

echo "SLURM_LOG_DIR: $SLURM_LOG_DIR"
echo "SLURM_COSMOS_CURATE_CONFIG_DIR: $SLURM_COSMOS_CURATE_CONFIG_DIR"
echo "SLURM_AWS_CREDS_DIR: $SLURM_AWS_CREDS_DIR"
echo "SLURM_AZURE_CREDS_DIR: $SLURM_AZURE_CREDS_DIR"
echo "SLURM_WORKSPACE: $SLURM_WORKSPACE"
echo "SLURM_IMAGE_DIR: $SLURM_IMAGE_DIR"
echo "SLURM_SOURCE_DIR: $SLURM_SOURCE_DIR"

echo "--------------------------------"
echo "Setting CONTAINER_MOUNTS..."

SLURM_COSMOS_CURATE_CONFIG_MOUNT="${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml:/cosmos_curate/config/cosmos_curate.yaml"
SLURM_WORKSPACE_MOUNT="${SLURM_WORKSPACE}:/config"
MOUNTS="${SLURM_COSMOS_CURATE_CONFIG_MOUNT},${SLURM_WORKSPACE_MOUNT}"

if [ -f "${SLURM_AWS_CREDS_DIR}/credentials" ]; then
    MOUNTS+=",${SLURM_AWS_CREDS_DIR}/credentials:/creds/s3_creds"
else
    echo "AWS credentials not found at ${SLURM_AWS_CREDS_DIR}/credentials; skipping S3 mount."
fi

if [ -f "${SLURM_AZURE_CREDS_DIR}/credentials" ]; then
    MOUNTS+=",${SLURM_AZURE_CREDS_DIR}/credentials:/creds/azure_creds"
else
    echo "Azure credentials not found at ${SLURM_AZURE_CREDS_DIR}/credentials; skipping Azure mount."
fi

export CONTAINER_MOUNTS="$MOUNTS"

echo "--------------------------------"
echo "CONTAINER_MOUNTS: $CONTAINER_MOUNTS"
