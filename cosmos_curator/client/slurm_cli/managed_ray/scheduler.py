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
"""Slurm submission, cancellation, and state queries issued from the login node.

The state names and the array-task liveliness rule these queries are read against live in
:mod:`.onnode.slurm_ray_state`, because head cleanup has to reduce a lane's tasks to one state the same way.
"""

import re
import shlex
from pathlib import Path
from typing import TypedDict

from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import (
    NONTERMINAL_SLURM_STATES,
    array_task_id,
    base_job_id,
    normalize_slurm_state,
    slurm_state_rank,
)
from cosmos_curator.client.slurm_cli.managed_ray.remote import (
    SlurmRayOperationError,
    run_remote,
)
from cosmos_curator.client.slurm_cli.slurm_submit import ConnectionProtocol

_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")
_SACCT_FIELD_COUNT = 5
_SQUEUE_FIELD_COUNT = 4


class SlurmJobState(TypedDict):
    """One observation of a Slurm job, merged from accounting and queue views.

    For a lane that renews through a job array this describes the lane as a whole. ``array_task_id`` identifies one
    concrete task when the selected scheduler record has one; it is never chronological progress.
    """

    state: str
    exit_code: str | None
    node_list: str | None
    reason: str | None
    restart_count: int | None
    array_task_id: int | None


def unknown_job_state(state: str = "UNKNOWN") -> SlurmJobState:
    """Return an observation carrying no detail beyond a state, for a job the scheduler did not describe."""
    return SlurmJobState(
        state=state, exit_code=None, node_list=None, reason=None, restart_count=None, array_task_id=None
    )


def _parse_sbatch_job_id(output: str) -> str:
    job_id = output.strip().split(";", maxsplit=1)[0]
    if not job_id or not _JOB_ID_PATTERN.fullmatch(job_id):
        msg = f"Could not parse a Slurm job ID from sbatch output: {output!r}"
        raise SlurmRayOperationError(msg)
    return job_id


def _find_job_by_exact_name(connection: ConnectionProtocol, job_name: str) -> str | None:
    result = run_remote(
        connection,
        f'squeue --noheader --user="$USER" --name={shlex.quote(job_name)} --format=%i',
        hide=True,
        warn=True,
    )
    if not result.ok:
        return None
    job_ids = list(dict.fromkeys(base_job_id(line.strip()) for line in result.stdout.splitlines() if line.strip()))
    if len(job_ids) == 1:
        return _parse_sbatch_job_id(job_ids[0])
    if len(job_ids) > 1:
        msg = f"Multiple Slurm jobs unexpectedly use exact name {job_name!r}: {', '.join(job_ids)}"
        raise SlurmRayOperationError(msg)
    return None


def submit_script(
    connection: ConnectionProtocol,
    path: Path,
    *,
    job_name: str,
    options: tuple[str, ...] = (),
    arguments: tuple[str, ...] = (),
) -> str:
    """Submit one batch script, recovering the job ID by exact name if the response is lost."""
    try:
        command = ["sbatch", "--parsable", f"--job-name={job_name}", *options, str(path), *arguments]
        result = run_remote(connection, shlex.join(command), hide=True)
        return _parse_sbatch_job_id(result.stdout)
    except Exception:
        recovered_job_id = _find_job_by_exact_name(connection, job_name)
        if recovered_job_id is not None:
            return recovered_job_id
        raise


def cancel_jobs(connection: ConnectionProtocol, job_ids: list[str]) -> list[str]:
    """Cancel each job in order and return exactly the IDs scancel accepted."""
    canceled: list[str] = []
    for job_id in job_ids:
        result = run_remote(connection, f"scancel --quiet {shlex.quote(job_id)}", hide=True, warn=True)
        if result.ok:
            canceled.append(job_id)
    return canceled


def query_job_states(connection: ConnectionProtocol, job_ids: list[str]) -> dict[str, SlurmJobState]:
    """Merge the accounting and queue views of every recorded job.

    A job absent from both is ``NOT_FOUND`` when squeue answered and ``UNKNOWN`` when it did not, so a failed
    query is never mistaken for a finished allocation.
    """
    if not job_ids:
        return {}
    for job_id in job_ids:
        if not _JOB_ID_PATTERN.fullmatch(job_id):
            msg = f"Manifest contains an invalid Slurm job ID: {job_id!r}"
            raise SlurmRayOperationError(msg)

    states = _accounting_job_states(connection, job_ids)
    queued = _queued_job_states(connection, job_ids)
    for job_id, queued_state in (queued or {}).items():
        accounting = states.get(job_id)
        if accounting is not None:
            queued_state["exit_code"] = accounting["exit_code"]
            queued_state["node_list"] = queued_state["node_list"] or accounting["node_list"]
            queued_state["restart_count"] = accounting["restart_count"]
        states[job_id] = queued_state

    absent = "UNKNOWN" if queued is None else "NOT_FOUND"
    for job_id in job_ids:
        states.setdefault(job_id, unknown_job_state(absent))
    return states


def is_nonterminal(state: SlurmJobState) -> bool:
    """Report whether an observed job may still hold an allocation."""
    return state["state"] in NONTERMINAL_SLURM_STATES


def _merge_job_state(existing: SlurmJobState | None, incoming: SlurmJobState) -> SlurmJobState:
    """Reduce the several records a renewing lane reports at once to its liveliest one."""
    if existing is None or slurm_state_rank(incoming["state"]) > slurm_state_rank(existing["state"]):
        return incoming
    # Array task order is undefined. Use the task ID only as a deterministic tie-breaker between equally live
    # scheduler records, never as an indication of which one ran later.
    if slurm_state_rank(incoming["state"]) == slurm_state_rank(existing["state"]) and (
        incoming["array_task_id"] or 0
    ) > (existing["array_task_id"] or 0):
        return incoming
    return existing


def _accounting_job_states(connection: ConnectionProtocol, job_ids: list[str]) -> dict[str, SlurmJobState]:
    result = run_remote(
        connection,
        "sacct -X --noheader --parsable2 "
        f"--jobs={shlex.quote(','.join(job_ids))} "
        # JobID rather than JobIDRaw: an array task's raw ID is an unrelated number, while JobID keeps it addressable
        # as <array id>_<task>, which is what the manifest can be matched against.
        "--format=JobID,State,ExitCode,NodeList,Restarts",
        hide=True,
        warn=True,
    )
    if not result.ok:
        return {}

    states: dict[str, SlurmJobState] = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < _SACCT_FIELD_COUNT:
            continue
        reported_id, state, exit_code, node_list, restarts = fields[:_SACCT_FIELD_COUNT]
        job_id = base_job_id(reported_id)
        if job_id not in job_ids:
            continue
        try:
            restart_count: int | None = int(restarts or "0")
        except ValueError:
            restart_count = None
        states[job_id] = _merge_job_state(
            states.get(job_id),
            SlurmJobState(
                state=normalize_slurm_state(state),
                exit_code=exit_code or None,
                node_list=node_list or None,
                reason=None,
                restart_count=restart_count,
                array_task_id=array_task_id(reported_id),
            ),
        )
    return states


def _queued_job_states(connection: ConnectionProtocol, job_ids: list[str]) -> dict[str, SlurmJobState] | None:
    """Return queued state for the requested jobs, or ``None`` when squeue could not be reached.

    Unlike sacct, squeue fails the whole query when any ``--jobs`` argument has already left the queue, which is
    normal here. List the user's jobs instead and filter locally so one finished lane cannot hide the others.
    """
    result = run_remote(
        connection,
        "squeue --noheader --user=\"$USER\" --format='%i|%T|%R|%N'",
        hide=True,
        warn=True,
    )
    if not result.ok:
        return None

    states: dict[str, SlurmJobState] = {}
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < _SQUEUE_FIELD_COUNT:
            continue
        reported_id, state, reason, node_list = fields[:_SQUEUE_FIELD_COUNT]
        job_id = base_job_id(reported_id)
        if job_id not in job_ids:
            continue
        normalized_state = normalize_slurm_state(state)
        states[job_id] = _merge_job_state(
            states.get(job_id),
            SlurmJobState(
                state=normalized_state,
                exit_code=None,
                node_list=node_list or None,
                reason=(reason or None) if normalized_state == "PENDING" else None,
                restart_count=None,
                array_task_id=array_task_id(reported_id),
            ),
        )
    return states
