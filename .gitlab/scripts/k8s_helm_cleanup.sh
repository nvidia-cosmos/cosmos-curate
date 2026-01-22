#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cleanup script for k8s_helm_test - called from after_script

set +e  # Don't exit on errors during cleanup

NAMESPACE="ci-helm-test-${CI_PIPELINE_ID:-local}"

echo "=== K8s Helm Test Cleanup ==="
echo "Deleting namespace ${NAMESPACE}..."

kubectl delete namespace "${NAMESPACE}" --ignore-not-found --wait=false || true

# Wait for namespace to be gone (up to 2 min)
for i in {1..24}; do
    if ! kubectl get namespace "${NAMESPACE}" 2>/dev/null; then
        echo "Namespace deleted"
        exit 0
    fi
    echo "Waiting for namespace deletion... ($i/24)"
    sleep 5
done

echo "WARNING: Namespace ${NAMESPACE} may still exist"
