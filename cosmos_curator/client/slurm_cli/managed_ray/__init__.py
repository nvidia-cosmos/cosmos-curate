# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Managed Ray clusters built from independent Slurm jobs.

``ray_cli`` is this package's only public surface. Everything else is internal to the launcher:

- :mod:`.cli` defines the ``cosmos-curator slurm ray`` command group
- :mod:`.config` resolves and validates the submission config
- :mod:`.lifecycle` implements submit, list, scale, and stop
- :mod:`.status` implements the combined manifest, Slurm, and Ray view
- :mod:`.remote` wraps the login-node connection and the private run directory
- :mod:`.scheduler` wraps sbatch, scancel, and the Slurm state queries
- :mod:`.render` resolves container paths and renders the batch scripts
- :mod:`.onnode` holds the modules uploaded into a run directory and executed on cluster nodes
"""

from cosmos_curator.client.slurm_cli.managed_ray.cli import ray_cli

__all__ = ["ray_cli"]
