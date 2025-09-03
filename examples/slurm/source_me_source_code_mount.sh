#!/bin/bash

# Validate required environment variables
for var in SLURM_AWS_CREDS_DIR SLURM_COSMOS_CURATE_CONFIG_DIR SLURM_WORKSPACE SLURM_SOURCE_DIR; do
    if [ -z "${!var}" ]; then
        echo "Error: $var is not defined. Please source source_me_env_vars.sh first."
        return
    fi
done

SLURM_AWS_CREDS_MOUNT="${SLURM_AWS_CREDS_DIR}/credentials:/creds/s3_creds"
SLURM_COSMOS_CURATE_CONFIG_MOUNT="${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml:/cosmos_curate/config/cosmos_curate.yaml"
SLURM_WORKSPACE_MOUNT="${SLURM_WORKSPACE}:/config"

echo "Adding source-code mount CONTAINER_MOUNTS..."
SLURM_SOURCE_MOUNT="${SLURM_SOURCE_DIR}/cosmos_curate/:/opt/cosmos-curate/cosmos_curate"
export CONTAINER_MOUNTS="${SLURM_AWS_CREDS_MOUNT},${SLURM_COSMOS_CURATE_CONFIG_MOUNT},${SLURM_WORKSPACE_MOUNT},${SLURM_SOURCE_MOUNT}"

echo "--------------------------------"
echo "CONTAINER_MOUNTS: $CONTAINER_MOUNTS"
