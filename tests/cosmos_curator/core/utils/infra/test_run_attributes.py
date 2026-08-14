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

"""Tests for ``cosmos_curator.core.utils.infra.run_attributes``."""

import os

import pytest

from cosmos_curator.core.utils.infra.run_attributes import (
    ENV_OTLP_RUN_ATTRIBUTES,
    ENV_OTLP_RUN_ATTRIBUTES_MAP,
    ENV_OTLP_RUN_ATTRIBUTES_VALUES,
    collect_run_attributes,
    otlp_run_attributes_enabled,
    set_otlp_run_attributes_enabled,
    short_host_label,
)


@pytest.fixture(autouse=True)
def _clear_run_attributes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        ENV_OTLP_RUN_ATTRIBUTES,
        ENV_OTLP_RUN_ATTRIBUTES_MAP,
        ENV_OTLP_RUN_ATTRIBUTES_VALUES,
        "SLURM_JOB_ID",
        "SLURM_JOBID",
        "SLURM_JOB_USER",
        "SLURM_JOB_NAME",
        "SLURM_RESTART_COUNT",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
        "SLURM_CLUSTER_NAME",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NNODES",
        "PRIMARY_NODE_HOSTNAME",
        "HEAD_NODE_ADDR",
        "SLURMD_NODENAME",
        "RAY_JOB_ID",
    ):
        monkeypatch.delenv(key, raising=False)


class TestOtlpRunAttributesToggle:
    """``otlp_run_attributes_enabled`` and ``set_otlp_run_attributes_enabled``."""

    def test_enabled_by_default(self) -> None:
        """Run attributes are on when the env var is unset."""
        assert otlp_run_attributes_enabled() is True

    def test_disabled_when_env_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``COSMOS_CURATOR_OTLP_RUN_ATTRIBUTES=0`` disables export."""
        monkeypatch.setenv(ENV_OTLP_RUN_ATTRIBUTES, "0")
        assert otlp_run_attributes_enabled() is False

    def test_set_disabled(self) -> None:
        """``set_otlp_run_attributes_enabled(enabled=False)`` writes ``0``."""
        set_otlp_run_attributes_enabled(enabled=False)
        assert os.environ.get(ENV_OTLP_RUN_ATTRIBUTES) == "0"

    def test_set_enabled_clears_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-enabling removes the disable marker from the environment."""
        monkeypatch.setenv(ENV_OTLP_RUN_ATTRIBUTES, "0")
        set_otlp_run_attributes_enabled(enabled=True)
        assert ENV_OTLP_RUN_ATTRIBUTES not in os.environ


class TestCollectRunAttributes:
    """``collect_run_attributes`` reads Slurm and local identity from env."""

    def test_slurm_labels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Standard Slurm job fields map to Prom SD-compatible keys."""
        monkeypatch.setenv("SLURM_JOB_ID", "80305")
        monkeypatch.setenv("SLURM_JOB_USER", "alice")
        monkeypatch.setenv("SLURM_JOB_NAME", "split-run")
        monkeypatch.setenv("SLURM_RESTART_COUNT", "1")
        attrs = collect_run_attributes()
        expected = {
            "slurm_job_id": "80305",
            "slurm_job_user": "alice",
            "slurm_job_name": "split-run",
            "slurm_restart_count": "1",
        }
        assert expected.items() <= attrs.items()

    def test_slurm_jobid_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SLURM_JOBID`` is used when ``SLURM_JOB_ID`` is absent."""
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setenv("SLURM_JOBID", "42")
        assert collect_run_attributes()["slurm_job_id"] == "42"

    def test_user_name_when_not_slurm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``user.name`` comes from ``USER`` when no Slurm user is set."""
        monkeypatch.setenv("USER", "bob")
        attrs = collect_run_attributes()
        assert attrs.get("user.name") == "bob"
        assert "slurm_job_user" not in attrs

    def test_slurm_user_precludes_user_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``slurm_job_user`` wins over ``USER`` on Slurm allocations."""
        monkeypatch.setenv("SLURM_JOB_USER", "alice")
        monkeypatch.setenv("USER", "bob")
        attrs = collect_run_attributes()
        assert attrs.get("slurm_job_user") == "alice"
        assert "user.name" not in attrs

    def test_extra_attributes_map_single_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Custom attribute map injects arbitrary env-backed labels."""
        monkeypatch.setenv("COSMOS_CUSTOMER", "acme")
        monkeypatch.setenv(
            ENV_OTLP_RUN_ATTRIBUTES_MAP,
            '{"customer":"COSMOS_CUSTOMER","mgmt_owner":"MISSING_ENV"}',
        )
        attrs = collect_run_attributes()
        assert attrs.get("customer") == "acme"
        assert "mgmt_owner" not in attrs

    def test_extra_attributes_map_env_fallback_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Map values can be fallback env name lists."""
        monkeypatch.setenv("SLURM_CLUSTER_NAME", "cluster-a")
        monkeypatch.setenv(
            ENV_OTLP_RUN_ATTRIBUTES_MAP,
            '{"cluster":["COSMOS_CLUSTER","SLURM_CLUSTER_NAME"]}',
        )
        attrs = collect_run_attributes()
        assert attrs.get("cluster") == "cluster-a"

    def test_extra_attributes_map_ignored_when_invalid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Malformed JSON map is ignored."""
        monkeypatch.setenv(ENV_OTLP_RUN_ATTRIBUTES_MAP, "{not-json")
        attrs = collect_run_attributes()
        assert "customer" not in attrs

    def test_extra_attributes_values_literal_map(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Literal attribute map injects caller-provided labels directly."""
        monkeypatch.setenv(
            ENV_OTLP_RUN_ATTRIBUTES_VALUES,
            '{"label_one":"value-one","label_two":"value-two"}',
        )
        attrs = collect_run_attributes()
        assert attrs.get("label_one") == "value-one"
        assert attrs.get("label_two") == "value-two"


class TestShortHostLabel:
    """``short_host_label`` normalizes host strings for OTLP resources."""

    def test_strips_domain(self) -> None:
        """Only the segment before the first dot is kept."""
        assert short_host_label("node03.cluster.example.com") == "node03"

    def test_empty_string(self) -> None:
        """An empty input returns an empty label."""
        assert short_host_label("") == ""
