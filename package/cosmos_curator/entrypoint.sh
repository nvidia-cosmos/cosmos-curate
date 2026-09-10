#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ "${COSMOS_CURATOR_TEE_STDOUT_STDERR:-}" = "1" ] || [ "${COSMOS_CURATOR_TEE_STDOUT_STDERR:-}" = "true" ]; then
    log_path="/tmp/curator/stdout-stderr.log"
    mkdir -p "$(dirname "$log_path")"
    touch "$log_path"
    chmod 600 "$log_path"
    # Process substitution, not a pipe: a pipe would make the exit status tee's.
    # Requires bash (see shebang) -- do not "simplify" to sh.
    exec > >(tee -a "$log_path") 2>&1
fi

exec "$@"
