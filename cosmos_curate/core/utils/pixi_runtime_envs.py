# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""A Pixi-based runtime environment for Ray."""

import copy

import attrs
import ray.runtime_env

from cosmos_xenna.ray_utils.runtime_envs import RuntimeEnv


@attrs.define
class PixiRuntimeEnv(RuntimeEnv):
    """RuntimeEnv subclass that uses Pixi to activate a specified Conda environment."""

    def to_ray_runtime_env(self) -> ray.runtime_env.RuntimeEnv:
        """Convert this PixiRuntimeEnv into a Ray RuntimeEnv.

        Setting environment variables and using pixi to run Python within the specified conda environment if provided.

        Returns:
            ray.runtime_env.RuntimeEnv: A Ray runtime environment with the specified settings.

        """
        env_vars = copy.deepcopy(self.extra_env_vars)
        if self.conda:
            py_executable = f"pixi run -e {self.conda.name} python"
            return ray.runtime_env.RuntimeEnv(env_vars=env_vars, py_executable=py_executable)
        return ray.runtime_env.RuntimeEnv(env_vars=env_vars)
