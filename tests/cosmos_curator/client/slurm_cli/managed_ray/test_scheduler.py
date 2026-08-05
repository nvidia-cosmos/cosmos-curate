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
"""Tests for the Slurm queries a managed Ray run issues from the login node."""

from pathlib import Path
from unittest.mock import Mock

from cosmos_curator.client.slurm_cli.managed_ray.onnode.slurm_ray_state import NONTERMINAL_SLURM_STATES
from cosmos_curator.client.slurm_cli.managed_ray.scheduler import (
    cancel_jobs,
    query_job_states,
    submit_script,
)
from tests.cosmos_curator.client.slurm_cli.managed_ray.launcher_stubs import FakeResult


def test_every_scheduler_call_is_bounded() -> None:
    """One unreachable login node must not hang a lifecycle command indefinitely."""
    connection = Mock()
    connection.run.return_value = FakeResult()

    assert cancel_jobs(connection, ["101", "102"]) == ["101", "102"]
    assert all(call.kwargs["timeout"] == 5 * 60 for call in connection.run.call_args_list)


def test_lost_sbatch_response_recovers_job_by_exact_name() -> None:
    """A unique deterministic job name recovers an accepted submission response."""
    connection = Mock()
    connection.run.side_effect = [
        RuntimeError("connection dropped after sbatch"),
        FakeResult(stdout="4321\n"),
    ]

    job_id = submit_script(connection, Path("/state/head.sbatch"), job_name="curator-cc-ray-deadbeef-head")

    assert job_id == "4321"
    assert "--name=curator-cc-ray-deadbeef-head" in connection.run.call_args_list[1].args[0]


def test_lost_sbatch_response_collapses_array_tasks_to_the_lane_job() -> None:
    """A running array task and its pending tail still identify one submitted lane."""
    connection = Mock()
    connection.run.side_effect = [
        RuntimeError("connection dropped after sbatch"),
        FakeResult(stdout="639573_41\n639573_[0-40%1]\n"),
    ]

    job_id = submit_script(connection, Path("/state/worker.sbatch"), job_name="curator-cc-ray-deadbeef-lane-0")

    assert job_id == "639573"


def test_renewing_lane_reports_its_live_array_task_not_its_finished_tasks() -> None:
    """A lane submitted as an array is as alive as its liveliest task."""
    connection = Mock()
    connection.run.side_effect = [
        # Two allocations already timed out; sacct reports each task separately.
        FakeResult(
            stdout="639573_0|TIMEOUT|0:0|pool0-1|0\n639573_1|TIMEOUT|0:0|pool0-2|0\n639573_2|RUNNING|0:0|pool0-9|0\n"
        ),
        # squeue shows the running task plus the collapsed pending tail.
        FakeResult(stdout="639573_2|RUNNING||pool0-9\n639573_[3-41%1]|PENDING|Priority|\n"),
    ]

    states = query_job_states(connection, ["639573"])

    assert states["639573"]["state"] == "RUNNING"
    assert states["639573"]["array_task_id"] == 2
    assert states["639573"]["node_list"] == "pool0-9"


def test_renewing_lane_stays_nonterminal_while_unused_array_tasks_are_queued() -> None:
    """Between allocations a lane has no running task, and must not look finished."""
    connection = Mock()
    connection.run.side_effect = [
        FakeResult(stdout="639573_0|TIMEOUT|0:0|pool0-1|0\n"),
        FakeResult(stdout="639573_[1-41%1]|PENDING|Priority|\n"),
    ]

    states = query_job_states(connection, ["639573"])

    assert states["639573"]["state"] == "PENDING"
    assert states["639573"]["state"] in NONTERMINAL_SLURM_STATES
    assert states["639573"]["reason"] == "Priority"
    # A collapsed range names several unordered tasks, not one chronological next allocation.
    assert states["639573"]["array_task_id"] is None


def test_exhausted_lane_is_terminal_once_no_allocation_remains() -> None:
    """A lane that used its whole budget reports terminal so scale can replace it."""
    connection = Mock()
    connection.run.side_effect = [
        FakeResult(stdout="639573_0|TIMEOUT|0:0|pool0-1|0\n639573_1|TIMEOUT|0:0|pool0-2|0\n"),
        FakeResult(stdout=""),
    ]

    states = query_job_states(connection, ["639573"])

    assert states["639573"]["state"] == "TIMEOUT"
    assert states["639573"]["state"] not in NONTERMINAL_SLURM_STATES
    # The task ID is only a deterministic identifier; it is not chronological progress.
    assert states["639573"]["array_task_id"] == 1


def test_decorated_slurm_states_reduce_to_a_bare_name() -> None:
    """Teardown cancels the lanes, and 'CANCELLED by <uid>' must not reach state matching or the operator."""
    connection = Mock()
    connection.run.side_effect = [
        FakeResult(stdout="2809479|CANCELLED by 25721|0:0|cpu-0003|0\n"),
        FakeResult(stdout=""),
    ]

    states = query_job_states(connection, ["2809479"])

    assert states["2809479"]["state"] == "CANCELLED"
    assert states["2809479"]["state"] not in NONTERMINAL_SLURM_STATES


def test_job_state_query_distinguishes_absent_jobs_from_query_failure() -> None:
    """A successful queue query can confirm that an unaccounted job is no longer active."""
    connection = Mock()
    connection.run.side_effect = [FakeResult(), FakeResult()]

    states = query_job_states(connection, ["100"])

    assert states["100"]["state"] == "NOT_FOUND"
    assert '--user="$USER"' in connection.run.call_args_list[1].args[0]

    connection.run.side_effect = [FakeResult(ok=False), FakeResult(ok=False)]
    states = query_job_states(connection, ["100"])

    assert states["100"]["state"] == "UNKNOWN"


def test_job_state_query_overlays_live_pending_reason() -> None:
    """Status retains the scheduler explanation for a pending allocation."""
    connection = Mock()
    connection.run.side_effect = [
        FakeResult(stdout="100|PENDING|0:0||0\n"),
        FakeResult(stdout="100|PENDING|Resources|\n"),
    ]

    states = query_job_states(connection, ["100"])

    assert states["100"] == {
        "state": "PENDING",
        "exit_code": "0:0",
        "node_list": None,
        "reason": "Resources",
        "restart_count": 0,
        "array_task_id": None,
    }
