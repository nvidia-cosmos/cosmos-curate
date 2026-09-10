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
"""Global pytest configuration."""

import os
import pathlib

import pytest

# Memray report readers may use debuginfod for native symbol lookup when
# this is inherited from the host, which can make profiling tests hang.
os.environ["DEBUGINFOD_URLS"] = ""


@pytest.fixture(scope="session")
def repo_root(pytestconfig: pytest.Config) -> pathlib.Path:
    """Return the checkout root, to pin the working directory of a spawned interpreter.

    Owned here rather than restated per test file because every import-purity
    probe depends on the same non-obvious asymmetry: ``python -c`` puts the
    child's working directory first on its ``sys.path``, while ``pythonpath = .``
    in ``pytest.ini`` anchors only the PARENT to the rootdir. A child left on an
    inherited working directory therefore resolves ``cosmos_curator`` to whatever
    copy that directory exposes - an installed one, in a checkout-plus-install
    environment - so a probe that does not pass this as ``cwd`` can report on
    code nobody edited.
    """
    return pytestconfig.rootpath
