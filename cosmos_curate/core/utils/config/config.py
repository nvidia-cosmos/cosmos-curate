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
"""Utilties for reading the config file.

See README.md for more info on what the config file is.
"""

from __future__ import annotations

from typing import Any

import attrs
import cattrs
import yaml

from cosmos_curate.core.utils.environment import CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE


@attrs.define
class OpenAI:
    """A class to represent and interact with OpenAI configuration details."""

    user: str = attrs.field(repr=True)
    api_key: str = attrs.field(repr=False)


@attrs.define
class Huggingface:
    """A class to represent and interact with Huggingface configuration details."""

    api_key: str = attrs.field(repr=False)


@attrs.define
class PostgresUser:
    """A class to represent and interact with Postgres database configuration details."""

    user: str
    password: str = attrs.field(repr=False)
    endpoint: str = attrs.field(repr=False)


@attrs.define
class Postgres:
    """A class to represent and interact with Postgres configuration details."""

    profiles: dict[str, PostgresUser] = attrs.Factory(dict)


@attrs.define
class ConfigFileData:
    """A class to handle the configuration data for cosmos-curate.

    This class supports loading from and saving to a configuration file.

    The config file is stored at ~/.config/cosmos_curate/config.yaml. An example config file is:

    ``` yaml
    huggingface:
        user: "abc"
        api_key: "xyz"
    postgres:
        profiles:
            nvc_dev:
               user: "abc"
               password: "xyz"
               endpoint: ""
    ```
    """

    open_ai: OpenAI | None = None
    huggingface: Huggingface | None = None
    postgres: Postgres | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfigFileData:
        """Convert a dict to a ConfigFileData object.

        Args:
            data: The dict to convert.

        Returns:
            The ConfigFileData object.

        """
        return cattrs.structure(data, cls)

    @classmethod
    def from_file(cls) -> ConfigFileData | None:
        """Load a ConfigFileData object from a file.

        Returns:
            The ConfigFileData object.

        """
        file = None
        if CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE.exists():
            file = CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE

        if file is None:
            return None

        return cls.from_dict(yaml.safe_load(file.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        """Convert a ConfigFileData object to a dict.

        Returns:
            The dict.

        """
        data = cattrs.unstructure(self, ConfigFileData)
        assert isinstance(data, dict)
        return data

    def get_postgres_profile(self, profile_name: str) -> PostgresUser:
        """Get a Postgres profile from the config file.

        Args:
            profile_name: The name of the profile to get.

        Returns:
            The PostgresUser object.

        """
        if self.postgres is None:
            error_msg = "Config does not have any Postgres data."
            raise ValueError(error_msg)
        if profile_name not in self.postgres.profiles:
            error_msg = (
                f"{profile_name=} not found in Postgres config. "
                f"Available profiles: {sorted(self.postgres.profiles.keys())}."
            )
            raise ValueError(error_msg)
        return self.postgres.profiles[profile_name]


def maybe_load_config() -> ConfigFileData | None:
    """Load a ConfigFileData object from a file.

    Returns:
        The ConfigFileData object.

    """
    return ConfigFileData.from_file()


def load_config() -> ConfigFileData:
    """Load a ConfigFileData object from a file.

    Returns:
        The ConfigFileData object.

    """
    config = maybe_load_config()
    if config is None:
        error_msg = (
            "cosmos-curate config file not found. "
            f"Please create a config file at {CONTAINER_PATHS_COSMOS_CURATOR_CONFIG_FILE}"
        )
        raise RuntimeError(error_msg)
    return config
