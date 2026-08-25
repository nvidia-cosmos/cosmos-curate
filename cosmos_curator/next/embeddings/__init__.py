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

"""Text / image / action feature embedders + preprocessing (Curator Next).

Only dependency-light contracts are re-exported here (schemas + model specs): the
model-backed embedder classes live in their submodules and defer torch, so the
package import stays CPU-safe (see tests/.../test_import_layers.py).
"""

from cosmos_curator.next.embeddings.model_specs import (
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_MODEL,
    TextModelSpec,
    VisionModelSpec,
    VisionPooling,
)
from cosmos_curator.next.embeddings.schemas import (
    ACTION_DIM,
    IMAGE_DIM,
    KEY_COLUMN,
    TEXT_DIM,
)

__all__ = [
    "ACTION_DIM",
    "DEFAULT_IMAGE_MODEL",
    "DEFAULT_TEXT_MODEL",
    "IMAGE_DIM",
    "KEY_COLUMN",
    "TEXT_DIM",
    "TextModelSpec",
    "VisionModelSpec",
    "VisionPooling",
]
