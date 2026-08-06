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

"""The ``--store-path`` surface shared by both data-integrity CLIs.

Kept apart from :mod:`.cli_common` for one reason: the store pulls in ``lance`` and
``pyarrow``, and a plain check should not pay for them. Everything here imports the
store lazily, inside the call, so the cost lands only on a run that actually asked
to persist.
"""

import argparse
from collections.abc import Mapping

from cosmos_curator.core.sensors.data_integrity.instruments import Thresholds
from cosmos_curator.core.sensors.data_integrity.results import StreamResult
from cosmos_curator.core.sensors.scripts._cli_cloud import CloudObjectStat


def add_store_args(parser: argparse.ArgumentParser) -> None:
    """Attach ``--store-path`` to ``parser``.

    Shared by both CLIs so the flag means one thing, and so a store written by
    ``di-check`` and one written by ``di-session`` are the same store.
    """
    parser.add_argument(
        "--store-path",
        default=None,
        metavar="PATH",
        help=(
            "Persist measurements and verdicts to a Lance store at this local directory or s3:// prefix "
            "(created if absent, appended to if not). Stored measurements can be re-judged under new "
            "thresholds without re-reading any source. Omit to check without saving anything."
        ),
    )


def persist_run(  # noqa: PLR0913 -- provenance plus credentials, all independent
    store_path: str,
    streams: list[StreamResult],
    *,
    session_path: str | None,
    thresholds: Thresholds,
    tool: str,
    cloud_stats: Mapping[str, CloudObjectStat] | None = None,
    s3_profile_name: str | None = None,
    azure_profile_name: str = "default",
    endpoint_url: str | None = None,
) -> str:
    """Append one run to the store at ``store_path`` and return its ``run_id``.

    Errors propagate: the operator asked for the results to be saved, so a store that
    could not be written is a failed run even when every check passed, and the CLIs
    turn that into exit code 2.
    """
    # Imported here, not at module scope, so `lance` / `pyarrow` load only for a run
    # that passed --store-path.
    from cosmos_curator.core.sensors.data_integrity import store  # noqa: PLC0415

    return store.write_run(
        store_path,
        streams,
        session_path=session_path,
        thresholds=thresholds,
        tool=tool,
        cloud_stats=cloud_stats,
        s3_profile_name=s3_profile_name,
        azure_profile_name=azure_profile_name,
        endpoint_url=endpoint_url,
    )
