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
"""CLI when you call a model file."""

import argparse

from loguru import logger

from cosmos_curate.core.utils.environment import (
    MODEL_WEIGHTS_PREFIX,
)
from cosmos_curate.core.utils.model_utils import (
    download_model_weights_from_huggingface_to_workspace,
    push_huggingface_model_to_cloud_storage,
)
from cosmos_curate.models.all_models import get_all_models


def setup_parsers() -> argparse.ArgumentParser:
    """Set up the parsers for the model CLI.

    Returns:
        The setup parser.

    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download and upload model weights",
    )

    subparsers = parser.add_subparsers(dest="command")

    download_parser = subparsers.add_parser(
        "download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Download model weights from huggingface to local workspace",
        help="Download model weights from huggingface to local workspace",
    )
    download_parser.set_defaults(func=_download)

    upload_parser = subparsers.add_parser(
        "upload",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Download model weights from huggingface and upload to S3 at prefix specified by --model-weights-prefix"
        ),
        help="Download model weights from huggingface and upload to S3 at prefix specified by --model-weights-prefix",
    )
    upload_parser.set_defaults(func=_upload)

    for x_parser in [download_parser, upload_parser]:
        x_parser.add_argument(
            "--models",
            type=str,
            default=",".join(x for x in get_all_models() if not x.startswith("_")),
            help=f"comma-separated list of models to download. Available models: {','.join(get_all_models().keys())}",
        )

    upload_parser.add_argument(
        "--model-weights-prefix",
        type=str,
        required=True,
        help=f"Cloud storage prefix for model weights, e.g. {MODEL_WEIGHTS_PREFIX}",
    )

    return parser


def main() -> None:
    """Run the model CLI.

    This function sets up the parsers and parses the arguments.
    """
    parser = setup_parsers()
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    args.func(args)


def _unpack_model_info(model: str) -> tuple[str, str | None, list[str] | None]:
    all_models = get_all_models()
    if model not in all_models:
        err_msg = f"Unknown model {model}. Available models: {','.join(all_models.keys())}"
        logger.error(err_msg)
        raise ValueError(err_msg)
    model_id = all_models[model]["model_id"]
    version = all_models[model]["version"]
    filelist = all_models[model]["filelist"]
    assert isinstance(model_id, str)
    assert isinstance(version, str) or version is None
    assert isinstance(filelist, list) or filelist is None
    return model_id, version, filelist


def _download(args: argparse.Namespace) -> None:
    models = args.models.split(",")
    for model in models:
        model_id, version, filelist = _unpack_model_info(model)
        download_model_weights_from_huggingface_to_workspace(
            model_id,
            version,
            filelist,
        )


def _upload(args: argparse.Namespace) -> None:
    models = args.models.split(",")
    for model in models:
        model_id, version, filelist = _unpack_model_info(model)
        push_huggingface_model_to_cloud_storage(
            model_id,
            version,
            filelist,
            model_weights_prefix=args.model_weights_prefix,
        )


if __name__ == "__main__":
    main()
