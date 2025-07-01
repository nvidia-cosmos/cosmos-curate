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
"""Secret handling for the benchmarks."""

import base64
import os
from typing import Any, cast

import attrs
import requests
import tenacity


def _get_secrets_from_env(env_vars: dict[str, str]) -> dict[str, str]:
    """Get secrets from environment variables."""
    missing = [var_name for var_name, env_name in env_vars.items() if os.getenv(env_name) is None]
    if missing:
        msg = f"Environment variables {', '.join(missing)} are not set"
        raise ValueError(msg)
    return {var_name: os.environ[env_name] for var_name, env_name in env_vars.items()}


@tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_exponential(multiplier=1, min=4, max=15))
def _get_bearer_token(url: str, api_key: str) -> str:
    """Get bearer token from Kratos.

    Args:
        url: URL of endpoint that supplies the bearer tokens.
        api_key: API key.

    Returns:
        Bearer token.

    """
    api_key_b64 = base64.b64encode(api_key.encode()).decode()
    headers = {"Content-Type": "application/x-www-form-urlencoded", "Authorization": f"Basic {api_key_b64}"}
    params = {"grant_type": "client_credentials", "scope": "telemetry-write"}
    response = requests.post(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    token_data = cast("dict[str, Any]", response.json())
    return cast("str", token_data["access_token"])


@attrs.define
class S3Secrets:
    """S3 secrets."""

    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str

    @classmethod
    def from_env(
        cls,
        aws_access_key_id_env: str,
        aws_secret_access_key_env: str,
        aws_region_env: str,
    ) -> "S3Secrets":
        """Get secrets from environment variables."""
        env_vars = {
            "aws_access_key_id": aws_access_key_id_env,
            "aws_secret_access_key": aws_secret_access_key_env,
            "aws_region": aws_region_env,
        }

        return cls(**_get_secrets_from_env(env_vars))


@attrs.define
class NvcfSecrets:
    """NVCF and AWS secrets."""

    ngc_org: str
    ngc_key: str

    @classmethod
    def from_env(
        cls,
        ngc_org_env: str,
        ngc_key_env: str,
    ) -> "NvcfSecrets":
        """Get secrets from environment variables."""
        env_vars = {
            "ngc_org": ngc_org_env,
            "ngc_key": ngc_key_env,
        }
        return cls(**_get_secrets_from_env(env_vars))


@attrs.define
class KratosSecrets:
    """Kratos secrets."""

    api_key: str
    bearer_token: str

    @classmethod
    def from_env(cls, api_key_env: str, url: str) -> "KratosSecrets":
        """Get secrets from environment variables.

        Args:
            api_key_env: API key environment variable.
            url: URL to get bearer token.

        Returns:
            KratosSecrets.

        """
        env_vars = {
            "api_key": api_key_env,
        }
        env_vals = _get_secrets_from_env(env_vars)

        api_key = env_vals["api_key"]  # guaranteed to be set by _get_secrets_from_env
        bearer_token = _get_bearer_token(url, api_key)

        return cls(api_key=api_key, bearer_token=bearer_token)
