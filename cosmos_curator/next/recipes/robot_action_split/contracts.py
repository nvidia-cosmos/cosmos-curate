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

"""Recipe-owned durable contract version constants for ``robot-action-split``."""

# Bump when non-configurable cut semantics change such that the same span_id
# would produce different output bytes (e.g. smart-cut → full-reencode switch).
MEDIA_CONTRACT_VERSION = 1

# Bump when the action binary format or field layout changes in a way that would
# produce different bytes for the same span/dataset (e.g. ACT2 header version bump,
# field reordering, or dtype change).
ACTION_CONTRACT_VERSION = 1

OUTCOME_RECORD_SCHEMA_VERSION = 2
RECEIPT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
LANCE_DATA_STORAGE_VERSION = "2.2"
