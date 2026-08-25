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

"""Torch device selection shared by the GPU embedder actors.

``torch`` is imported lazily inside the function so this helper (and the pure
embedder helpers that import it) stay importable without torch for CPU unit
tests and the import-layer checks.
"""


def resolve_torch_device(device: str | None = None) -> str:
    """Return the requested torch device, defaulting to cuda when available else cpu.

    Args:
        device: Explicit device string, or ``None`` to auto-select.

    Returns:
        ``device`` if given, otherwise ``"cuda"`` when a GPU is visible else
        ``"cpu"``.

    """
    if device is not None:
        return device
    import torch  # noqa: PLC0415 - deferred so this module imports without torch

    return "cuda" if torch.cuda.is_available() else "cpu"
