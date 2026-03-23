#!/bin/bash

RCLONE_REMOTE=":sftp,host=my-slurm-login-01.my-cluster.com:"

if [ -z "${SLURM_IMAGE_DIR}" ]; then
    echo "Error: SLURM_IMAGE_DIR is not defined"
elif [ -z "${COSMOS_CURATE_IMAGE_NAME}" ]; then
    echo "Error: COSMOS_CURATE_IMAGE_NAME is not defined"
elif [ ! -f "./${COSMOS_CURATE_IMAGE_NAME}" ]; then
    echo "Error: Image ./${COSMOS_CURATE_IMAGE_NAME} not found"
else
    echo "uploading image"
    ssh my-slurm-login-01.my-cluster.com mkdir -p "${SLURM_IMAGE_DIR}"
    rclone copyto -P "./${COSMOS_CURATE_IMAGE_NAME}" "${RCLONE_REMOTE}${SLURM_IMAGE_DIR}/${COSMOS_CURATE_IMAGE_NAME}"
fi
