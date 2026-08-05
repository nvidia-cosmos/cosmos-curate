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
"""CLI commands for managed Ray clusters backed by independent Slurm jobs."""

import json
import sys
from pathlib import Path
from typing import Annotated, NoReturn, cast

import invoke
import typer
import yaml
from pydantic import ValidationError
from typer import Argument, Option

from cosmos_curator.client.slurm_cli.managed_ray.config import (
    STATE_DIR_ENV_VAR,
    SlurmRayConfigError,
    default_state_dir,
    resolve_slurm_ray_config,
    slurm_ray_config_template,
    slurm_ray_config_to_json,
    slurm_ray_schema_json,
    slurm_ray_template_yaml,
)
from cosmos_curator.client.slurm_cli.managed_ray.lifecycle import (
    SlurmRayPartialOperationError,
    list_slurm_ray_runs,
    scale_slurm_ray_run,
    stop_slurm_ray_run,
    submit_slurm_ray_run,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import SlurmRayOperationError
from cosmos_curator.client.slurm_cli.managed_ray.status import status_slurm_ray_run
from cosmos_curator.client.slurm_cli.slurm_common import _DEFAULT_LOGIN_NODE, _get_username

_STATE_DIR_HELP = (
    "Shared directory holding run state. Must match the submit invocation; "
    f"set the launcher-wide default with ${{{STATE_DIR_ENV_VAR}}}."
)

# Options every lifecycle command takes, declared once so their help text cannot drift apart.
StateDirOption = Annotated[str, Option(help=_STATE_DIR_HELP)]
LoginNodeOption = Annotated[str, Option(help="Hostname of the Slurm login node. Defaults to local submission.")]
UsernameOption = Annotated[str, Option(help="Cluster username.")]
JsonOption = Annotated[bool, Option("--json", help="Emit machine-readable output.")]
OverridesOption = Annotated[
    list[str] | None,
    Option("--set", help="Resolved-config override in dotted PATH=VALUE form."),
]
ConfigArgument = Annotated[Path, Argument(help="Path to a JSON/YAML Slurm-Ray config.")]
RunIdArgument = Annotated[str, Argument(help="Managed Ray run ID.")]

_DEFAULT_STATE_DIR = default_state_dir()
_DEFAULT_USERNAME = _get_username()

# Everything a command can fail with once its arguments have parsed, reported through `_fail` rather than as a
# traceback: errors the launcher raises deliberately, plus the transport and filesystem it cannot speak for. A
# bare TypeError or ValueError is a launcher bug and should keep its traceback.
_CONFIG_ERRORS = (OSError, SlurmRayConfigError, ValidationError, yaml.YAMLError)
_LIFECYCLE_ERRORS = (*_CONFIG_ERRORS, SlurmRayOperationError, invoke.exceptions.Failure)

ray_cli = typer.Typer(
    help="Run-scoped Ray clusters built from independent Slurm jobs.",
    no_args_is_help=True,
)


@ray_cli.command()
def template(
    *,
    json_output: Annotated[bool, Option("--json", help="Emit the template as JSON.")] = False,
) -> None:
    """Print an editable submission config.

    The generated config names the schema version it was written for.
    """
    if json_output:
        typer.echo(json.dumps(slurm_ray_config_template(), indent=2))
    else:
        sys.stdout.write(slurm_ray_template_yaml())


@ray_cli.command()
def schema() -> None:
    """Print JSON Schema for the submission config."""
    sys.stdout.write(slurm_ray_schema_json())


@ray_cli.command(no_args_is_help=True)
def validate(
    config: ConfigArgument,
    *,
    set_overrides: OverridesOption = None,
    json_output: JsonOption = False,
) -> None:
    """Resolve and validate a submission config without connecting to Slurm."""
    try:
        resolved = resolve_slurm_ray_config(config, overrides=set_overrides or [])
    except _CONFIG_ERRORS as exc:
        _fail("invalid_config", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps({"ok": True, "config": resolved.model_dump(mode="json")}, indent=2))
    else:
        typer.echo("valid")


@ray_cli.command(no_args_is_help=True)
def render(
    config: ConfigArgument,
    *,
    set_overrides: OverridesOption = None,
) -> None:
    """Print the canonical resolved JSON config used for submission."""
    try:
        resolved = resolve_slurm_ray_config(config, overrides=set_overrides or [])
    except _CONFIG_ERRORS as exc:
        _fail("invalid_config", exc, json_output=False)
    sys.stdout.write(slurm_ray_config_to_json(resolved))


@ray_cli.command(no_args_is_help=True, context_settings={"allow_extra_args": True})
def submit(  # noqa: PLR0913
    config: ConfigArgument,
    command: Annotated[list[str], Argument(help="Pipeline command to run after '--'.")],
    *,
    set_overrides: OverridesOption = None,
    state_dir: StateDirOption = _DEFAULT_STATE_DIR,
    login_node: LoginNodeOption = _DEFAULT_LOGIN_NODE,
    username: UsernameOption = _DEFAULT_USERNAME,
    json_output: JsonOption = False,
) -> None:
    """Submit one managed Ray head and one requeueable job per worker lane."""
    try:
        resolved = resolve_slurm_ray_config(config, overrides=set_overrides or [])
        result = submit_slurm_ray_run(
            resolved,
            command,
            login_node=login_node,
            username=username,
            state_dir=state_dir,
        )
    except _LIFECYCLE_ERRORS as exc:
        _fail("submit_failed", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(result.json_payload(), indent=2))
    else:
        typer.echo(f"Run ID: {result.run_id}")
        typer.echo(f"Slurm cluster: {result.slurm_cluster_name}")
        typer.echo(f"Head job: {result.head_job_id}")
        typer.echo(f"Worker jobs: {', '.join(result.lane_job_ids)}")
        typer.echo(f"Manifest: {result.manifest_path}")
        typer.echo(f"Logs: {result.log_dir}")


@ray_cli.command(no_args_is_help=True)
def scale(  # noqa: PLR0913
    run_id: RunIdArgument,
    *,
    workers: Annotated[int, Option("--workers", min=0, help="Target number of nonterminal worker lanes.")],
    state_dir: StateDirOption = _DEFAULT_STATE_DIR,
    login_node: LoginNodeOption = _DEFAULT_LOGIN_NODE,
    username: UsernameOption = _DEFAULT_USERNAME,
    json_output: JsonOption = False,
) -> None:
    """Scale a run to a target number of worker lanes."""
    try:
        result = scale_slurm_ray_run(
            run_id,
            worker_lanes=workers,
            login_node=login_node,
            username=username,
            state_dir=state_dir,
        )
    except _LIFECYCLE_ERRORS as exc:
        _fail("scale_failed", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(result.json_payload(), indent=2))
        return
    typer.echo(
        f"Run {result.run_id}: {result.worker_lanes} worker lanes "
        f"(previously {result.previous_worker_lanes}, target {result.target_worker_lanes})"
    )
    typer.echo(f"Submitted jobs: {_format_job_ids(result.submitted_job_ids)}")
    typer.echo(f"Canceled jobs: {_format_job_ids(result.canceled_job_ids)}")


@ray_cli.command(name="list")
def list_runs(
    *,
    state_dir: StateDirOption = _DEFAULT_STATE_DIR,
    login_node: LoginNodeOption = _DEFAULT_LOGIN_NODE,
    username: UsernameOption = _DEFAULT_USERNAME,
    json_output: JsonOption = False,
) -> None:
    """List every managed Ray run under a state directory."""
    try:
        runs = list_slurm_ray_runs(login_node=login_node, username=username, state_dir=state_dir)
    except _LIFECYCLE_ERRORS as exc:
        _fail("list_failed", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps({"runs": runs}, indent=2))
        return
    if not runs:
        typer.echo(f"No managed Ray runs under {state_dir}")
        return
    for run in runs:
        lanes = "?" if run["recorded_worker_lanes"] is None else run["recorded_worker_lanes"]
        cluster = run["slurm_cluster_name"] or "?"
        typer.echo(
            f"{run['run_id']}  {run['state']:<11}  cluster={cluster}  "
            f"recorded_lanes={lanes}  {run['created_at'] or '-'}"
        )


@ray_cli.command(no_args_is_help=True)
def status(
    run_id: RunIdArgument,
    *,
    state_dir: StateDirOption = _DEFAULT_STATE_DIR,
    login_node: LoginNodeOption = _DEFAULT_LOGIN_NODE,
    username: UsernameOption = _DEFAULT_USERNAME,
    json_output: JsonOption = False,
) -> None:
    """Report manifest, Slurm allocation, and timestamped Ray state."""
    try:
        result = status_slurm_ray_run(
            run_id,
            login_node=login_node,
            username=username,
            state_dir=state_dir,
        )
    except _LIFECYCLE_ERRORS as exc:
        _fail("status_failed", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    typer.echo(f"Run {result['run_id']} on {result['slurm_cluster_name']}: {result['state']}")
    head = result["head"]
    head_reason = f", reason={head['reason']}" if head["reason"] else ""
    typer.echo(f"Head {head['job_id']}: {head['state']}{head_reason}")
    typer.echo(f"Driver: {result['driver_state']}")
    for lane in result["lanes"]:
        restart_count = lane["restart_count"]
        restart_text = "unknown" if restart_count is None else str(restart_count)
        reason_text = f", reason={lane['reason']}" if lane["reason"] else ""
        array_task_id = lane["array_task_id"]
        array_task_text = (
            f", array_task={array_task_id}" if lane["state"] == "RUNNING" and array_task_id is not None else ""
        )
        typer.echo(
            f"Lane {lane['lane']} ({lane['job_id']}): {lane['state']}"
            f"{array_task_text}, restarts={restart_text}{reason_text}"
        )
    ray_address = result["ray"]["address"] or "unknown"
    typer.echo(f"Ray: {result['ray']['state']} ({ray_address})")
    typer.echo(f"Logs: {result['log_dir']}")


@ray_cli.command(no_args_is_help=True)
def stop(
    run_id: RunIdArgument,
    *,
    state_dir: StateDirOption = _DEFAULT_STATE_DIR,
    login_node: LoginNodeOption = _DEFAULT_LOGIN_NODE,
    username: UsernameOption = _DEFAULT_USERNAME,
    json_output: JsonOption = False,
) -> None:
    """Cancel the exact worker and head job IDs recorded for one run."""
    try:
        result = stop_slurm_ray_run(
            run_id,
            login_node=login_node,
            username=username,
            state_dir=state_dir,
        )
    except _LIFECYCLE_ERRORS as exc:
        _fail("stop_failed", exc, json_output=json_output)

    if json_output:
        typer.echo(json.dumps(result.json_payload(), indent=2))
        return
    typer.echo(f"Run {result.run_id}: {result.state}")
    typer.echo(f"Canceled worker jobs: {_format_job_ids(result.canceled_lane_job_ids)}")
    typer.echo(f"Canceled head jobs: {_format_job_ids(result.canceled_head_job_ids)}")


def _format_job_ids(job_ids: list[str]) -> str:
    return ", ".join(job_ids) or "none"


def _error_details(exc: Exception) -> list[dict[str, object]]:
    if isinstance(exc, ValidationError):
        return cast("list[dict[str, object]]", json.loads(exc.json(include_url=False)))
    if isinstance(exc, SlurmRayConfigError):
        return [{"operation": "config_resolution", "message": str(exc)}]
    if isinstance(exc, SlurmRayPartialOperationError):
        return [{"operation": "lifecycle", "message": str(exc), "result": exc.result}]
    return [{"operation": "command", "message": str(exc)}]


def _fail(code: str, exc: Exception, *, json_output: bool) -> NoReturn:
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "ok": False,
                    "error": code,
                    "message": str(exc),
                    "details": _error_details(exc),
                },
                indent=2,
            ),
            err=True,
        )
    else:
        typer.echo(str(exc), err=True)
        if isinstance(exc, SlurmRayPartialOperationError):
            typer.echo(f"Partial result: {json.dumps(exc.result, sort_keys=True)}", err=True)
    raise typer.Exit(2)


if __name__ == "__main__":
    ray_cli()
