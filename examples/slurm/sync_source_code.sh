#!/bin/bash

# get directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

if [ -z "${SLURM_SOURCE_DIR}" ]; then
    echo "Error: SLURM_SOURCE_DIR is not defined"
else
    echo "sync-ing source code"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_SOURCE_DIR}/cosmos_curate/"
    rsync -avh --exclude="*.pyc" --exclude="__pycache__/" \
        "${ROOT_DIR}/cosmos_curate/" \
        "my-slurm-login-01.my-cluster.com:${SLURM_SOURCE_DIR}/cosmos_curate/"
fi

