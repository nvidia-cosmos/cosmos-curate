#!/bin/bash

RCLONE_REMOTE=":sftp,host=my-slurm-login-01.my-cluster.com:"

if [ -z "${SLURM_COSMOS_CURATE_CONFIG_DIR}" ]; then
    echo "Error: SLURM_COSMOS_CURATE_CONFIG_DIR is not defined"
else
    echo "sync-ing cosmos_curate config yaml"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_COSMOS_CURATE_CONFIG_DIR}"
    rclone copyto -P ~/.config/cosmos_curate/config.yaml "${RCLONE_REMOTE}${SLURM_COSMOS_CURATE_CONFIG_DIR}/config.yaml"
fi

if [ -z "${SLURM_AWS_CREDS_DIR}" ]; then
    echo "SLURM_AWS_CREDS_DIR is not defined, skipping AWS creds sync"
elif [ -f ~/.aws/credentials ]; then
    echo "sync-ing aws creds"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_AWS_CREDS_DIR}"
    rclone copyto -P ~/.aws/credentials "${RCLONE_REMOTE}${SLURM_AWS_CREDS_DIR}/credentials"
else
    echo "AWS credentials file ~/.aws/credentials not found, skipping AWS creds sync"
fi

if [ -z "${SLURM_AZURE_CREDS_DIR}" ]; then
    echo "SLURM_AZURE_CREDS_DIR is not defined, skipping Azure creds sync"
elif [ -f ~/.azure/credentials ]; then
    echo "sync-ing azure creds"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_AZURE_CREDS_DIR}"
    rclone copyto -P ~/.azure/credentials "${RCLONE_REMOTE}${SLURM_AZURE_CREDS_DIR}/credentials"
else
    echo "Azure credentials file ~/.azure/credentials not found, skipping Azure creds sync"
fi

if [ -z "${SLURM_WORKSPACE}" ]; then
    echo "Error: SLURM_WORKSPACE is not defined"
else
    echo "sync-ing models"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_WORKSPACE}/models"
    rclone copy -P "${COSMOS_CURATE_LOCAL_WORKSPACE_PREFIX:-$HOME}/cosmos_curate_local_workspace/models/" \
        "${RCLONE_REMOTE}${SLURM_WORKSPACE}/models/"
fi
