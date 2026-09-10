#!/usr/bin/env python3
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

"""Secret extractor script for init container.

Extracts certificate and key from NVCF secrets JSON and writes them to the appropriate location.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import NoReturn

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("secret_extractor")

SECRETS_FILE = Path(os.getenv("SECRET_EXTRACTOR_SECRETS_FILE", "/var/secrets/secrets.json"))
OUTPUT_DIR = Path(os.getenv("SECRET_EXTRACTOR_OUTPUT_DIR", "/etc/curator-remote-write/certs"))
CERT_FILE = OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_CERT_FILE", "tls.crt")
KEY_FILE = OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_KEY_FILE", "tls.key")
CA_FILE = OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_CA_FILE", "ca.crt")
SECONDARY_OUTPUT_DIR_VALUE = os.getenv("SECRET_EXTRACTOR_SECONDARY_OUTPUT_DIR", "")
SECONDARY_OUTPUT_DIR = Path(SECONDARY_OUTPUT_DIR_VALUE) if SECONDARY_OUTPUT_DIR_VALUE else None

# Secret keys in JSON
CERT_KEY = os.getenv("SECRET_EXTRACTOR_CERT_KEY", "byo-metrics-receiver-client-crt")
KEY_KEY = os.getenv("SECRET_EXTRACTOR_KEY_KEY", "byo-metrics-receiver-client-key")
CA_KEY = os.getenv("SECRET_EXTRACTOR_CA_KEY", "")
SECONDARY_CERT_KEY = os.getenv("SECRET_EXTRACTOR_SECONDARY_CERT_KEY", "")
SECONDARY_KEY_KEY = os.getenv("SECRET_EXTRACTOR_SECONDARY_KEY_KEY", "")
SECONDARY_CA_KEY = os.getenv("SECRET_EXTRACTOR_SECONDARY_CA_KEY", "")

# Timing constants
MAX_WAIT_TIME = 300  # 5 minutes in seconds
WAIT_INTERVAL = 5  # Check every 5 seconds


def wait_for_secrets_file() -> None:
    """Wait for secrets file to appear, up to MAX_WAIT_TIME seconds."""
    deadline = time.time() + MAX_WAIT_TIME

    while time.time() < deadline:
        if SECRETS_FILE.exists():
            return
        logger.info("Waiting for secrets file %s to appear...", SECRETS_FILE)
        time.sleep(WAIT_INTERVAL)

    error_msg = f"Secrets file {SECRETS_FILE} did not appear within {MAX_WAIT_TIME} seconds"
    logger.error(error_msg)
    raise TimeoutError(error_msg)


def read_secrets() -> tuple[dict[str, str], str, str, str | None]:
    """Read and validate secrets from JSON file."""
    try:
        secrets = json.loads(SECRETS_FILE.read_text())
    except json.JSONDecodeError as err:
        # Don't include decode error as it might contain parts of the secret
        error_msg = f"Failed to parse {SECRETS_FILE} as JSON"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err
    except OSError as err:
        error_msg = f"Failed to read {SECRETS_FILE}: {err}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err

    missing = [key for key in (CERT_KEY, KEY_KEY) if not secrets.get(key)]
    if missing:
        error_msg = f"Missing or empty required secrets: {', '.join(missing)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    ca = secrets.get(CA_KEY) if CA_KEY else None
    if CA_KEY and not ca:
        error_msg = f"Missing or empty configured CA secret: {CA_KEY}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return secrets, secrets[CERT_KEY], secrets[KEY_KEY], ca


def write_secrets(cert: str, key: str, ca: str | None) -> None:
    """Write certificate and key to output directory."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CERT_FILE.write_text(cert)
        KEY_FILE.write_text(key)
        KEY_FILE.chmod(0o600)
        if ca:
            CA_FILE.write_text(ca)
    except OSError as err:
        error_msg = f"Failed to write secrets to {OUTPUT_DIR}: {err}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err


def write_secondary_secrets(secrets: dict[str, str]) -> None:
    """Write an optional second certificate set when all required keys are present."""
    if not SECONDARY_OUTPUT_DIR or not SECONDARY_CERT_KEY or not SECONDARY_KEY_KEY:
        return

    cert = secrets.get(SECONDARY_CERT_KEY)
    key = secrets.get(SECONDARY_KEY_KEY)
    if not cert or not key:
        logger.warning("Optional secondary cert secrets are missing; skipping %s", SECONDARY_OUTPUT_DIR)
        return

    ca = secrets.get(SECONDARY_CA_KEY) if SECONDARY_CA_KEY else None
    if SECONDARY_CA_KEY and not ca:
        logger.warning("Optional secondary CA secret %s is missing; continuing without CA file", SECONDARY_CA_KEY)
    try:
        SECONDARY_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (SECONDARY_OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_SECONDARY_CERT_FILE", "tls.crt")).write_text(cert)
        secondary_key_file = SECONDARY_OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_SECONDARY_KEY_FILE", "tls.key")
        secondary_key_file.write_text(key)
        secondary_key_file.chmod(0o600)
        if ca:
            (SECONDARY_OUTPUT_DIR / os.getenv("SECRET_EXTRACTOR_SECONDARY_CA_FILE", "ca.crt")).write_text(ca)
    except OSError as err:
        error_msg = f"Failed to write secondary secrets to {SECONDARY_OUTPUT_DIR}: {err}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from err


def main() -> NoReturn:
    """Extract secrets from NVCF secrets JSON and write them to the appropriate location."""
    logger.info("Starting secret extraction process...")
    wait_for_secrets_file()
    logger.info("Found secrets file %s", SECRETS_FILE)

    secrets, cert, key, ca = read_secrets()
    logger.info("Successfully read secrets")

    write_secrets(cert, key, ca)
    write_secondary_secrets(secrets)
    logger.info("Successfully wrote secrets to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
