#!/bin/bash

if [ -z "${SLURM_COSMOS_CURATE_CONFIG_DIR}" ]; then
    echo "Error: SLURM_COSMOS_CURATE_CONFIG_DIR is not defined"
else
    echo "sync-ing cosmos_curate config yaml"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_COSMOS_CURATE_CONFIG_DIR}"
    rsync -avh ~/.config/cosmos_curate/config.yaml "my-slurm-login-01.my-cluster.com:${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml"
fi

if [ -z "${SLURM_AWS_CREDS_DIR}" ]; then
    echo "Error: SLURM_AWS_CREDS_DIR is not defined"
else
    echo "sync-ing aws creds"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_AWS_CREDS_DIR}"
    rsync -avh ~/.aws/credentials "my-slurm-login-01.my-cluster.com:${SLURM_AWS_CREDS_DIR}/credentials"
fi

if [ -z "${SLURM_WORKSPACE}" ]; then
    echo "Error: SLURM_WORKSPACE is not defined"
else
    echo "sync-ing models"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_WORKSPACE}/models"
    rsync -avh --progress "${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/models/" \
        "my-slurm-login-01.my-cluster.com:${SLURM_WORKSPACE}/models/"
fi
