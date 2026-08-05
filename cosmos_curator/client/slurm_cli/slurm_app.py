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
"""Slurm CLI app."""

import typer

from cosmos_curator.client.slurm_cli.managed_ray import ray_cli
from cosmos_curator.client.slurm_cli.slurm_shell import import_image_cli, shell_cli
from cosmos_curator.client.slurm_cli.slurm_submit import job_log_cli, submit_cli

slurm_cli = typer.Typer(
    context_settings={
        "max_content_width": 120,
    },
    pretty_exceptions_enable=False,
    no_args_is_help=True,
)

slurm_cli.add_typer(ray_cli, name="ray")
slurm_cli.command("submit", no_args_is_help=True)(submit_cli)
slurm_cli.command("shell")(shell_cli)
slurm_cli.command("import-image", no_args_is_help=True)(import_image_cli)
slurm_cli.command("job-log", no_args_is_help=True)(job_log_cli)
