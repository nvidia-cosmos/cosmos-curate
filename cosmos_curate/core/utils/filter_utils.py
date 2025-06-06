# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Filer utilities."""


def accept(_: object) -> bool:
    """Regardless of what is passed in, return True.

    Example usage:

    ```
    def do_something(data: list[str], filter_func: Callable[str, bool] | None = None)
        _filter_func = filter.accept if filter_func is None else filter_func
        [x for x in my_list if _filter_func(x)]
    ```
    """
    return True
