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
"""Utilities for writing models and managing the weights of those models.

See RAY_PIPELINES.md for an introduction to our model packaging system.
"""

import concurrent.futures
import hashlib
import json
import pathlib
import shutil
import time

import attrs
import huggingface_hub
import ray
import ray.util
from loguru import logger

from cosmos_curate.core.cf.nvcf_utils import (
    download_model_weights_from_nvcf_to_workspace,
    is_nvcf_container_deployment,
    is_nvcf_helm_deployment,
)
from cosmos_curate.core.utils import environment, storage_client, storage_utils
from cosmos_curate.core.utils.config import load_config
from cosmos_curate.core.utils.runtime import hardware_info, operation_utils, ray_cluster_utils
from cosmos_curate.core.utils.storage_client import DOWNLOAD_CHUNK_SIZE_BYTES

CUDA_DEVICE: str = "cuda:0"
_MODEL_DOWNLOADER_CPU_REQUEST: float = 1.0


@ray.remote(num_cpus=_MODEL_DOWNLOADER_CPU_REQUEST, runtime_env={"conda": "model_download"})
class _ModelWeightsDownloader:
    """A Ray actor class for downloading model weights on a specific node.

    This class is responsible for downloading model weights from various sources
    (NVCF, cloud storage, or local storage) to the workspace on a specific node.
    """

    def __init__(self) -> None:
        """Initialize the _ModelWeightsDownloader with the current node name."""
        self._node_name = ray_cluster_utils.get_node_name()

    def download_model_weights(self, model_names: list[str], source: str) -> tuple[str, dict[str, str]]:
        """Download model weights from the specified source.

        Args:
            model_names: List of model names to download.
            source: Source of the model weights (e.g., "nvcf-container", "nvcf-helm", or a storage path).

        Returns:
            A tuple containing:
                - The node name where the download was attempted
                - A dictionary of missing models and their paths (if any)

        """
        logger.info(f"Syncing model weights from {source} on {self._node_name}.")
        if source == "nvcf-container":
            logger.info("Using models auto-downloaed by NVCF")
        elif source == "nvcf-helm":
            download_model_weights_from_nvcf_to_workspace(
                list(model_names),
                download_dir=environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR,
            )
        elif storage_utils.is_remote_path(source):
            download_model_weights_from_cloud_storage_to_workspace(
                list(model_names), cloud_storage_model_weights_prefix=source
            )
        else:
            download_model_weights_from_local_to_workspace(list(model_names), local_model_weights_path=source)
        logger.complete()
        # final verification
        return self._check_missing_model_weights(model_names)

    def _check_missing_model_weights(self, model_names: list[str]) -> tuple[str, dict[str, str]]:
        """Check which model weights are missing after download attempt.

        Args:
            model_names: List of model names to check.

        Returns:
            A tuple containing:
                - The node name where the check was performed
                - A dictionary of missing models and their expected paths

        """
        missing_models = {}
        for model_name in model_names:
            model_path = environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / model_name
            if not model_path.exists():
                missing_models[model_name] = str(model_path)
        return self._node_name, missing_models


def download_model_weights_on_all_nodes(model_names: list[str], model_weight_prefix: str, num_nodes: int) -> None:
    """Download model weights on all nodes in the Ray cluster.

    This function determines the appropriate source for model weights based on the deployment
    environment (NVCF container or NVCF helm) and initiates the download process on all nodes.
    The download process runs in a separate process to isolate Ray's runtime environment.

    Args:
        model_names: List of model names to download.
        model_weight_prefix: Prefix for the model weights in cloud or local storage.
        num_nodes: Number of nodes in the Ray cluster to download the model weights on.

    """
    zero_start = time.time()
    source: str | None = None

    if is_nvcf_container_deployment():
        source = "nvcf-container"
    elif is_nvcf_helm_deployment():
        source = "nvcf-helm"
    else:
        source = model_weight_prefix

    bundles = [{"CPU": _MODEL_DOWNLOADER_CPU_REQUEST} for _ in range(num_nodes)]
    pg = ray.util.placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    executors = [_ModelWeightsDownloader.options(placement_group=pg).remote() for _ in range(num_nodes)]  # type: ignore[attr-defined]
    results = ray.get([x.download_model_weights.remote(model_names, source=source) for x in executors])
    for node_name, missing_models in results:
        if len(missing_models.keys()) > 0:
            for model_name, model_path in missing_models.items():
                logger.error(f"Model weights {model_name} not found on node {node_name} at {model_path}.")
    download_time = (time.time() - zero_start) / 60

    logger.info(f"---- Finished downloading model weights on all nodes in {download_time:.2f} minutes ----")


def hash_attrs_object(obj: attrs.AttrsInstance) -> str:
    """Hash an attrs object.

    Args:
        obj: The attrs object to hash.

    Returns:
        The hash of the attrs object.

    """
    json_string = json.dumps(attrs.asdict(obj), sort_keys=True).encode("utf-8")
    hash_object = hashlib.sha256(json_string)
    return hash_object.hexdigest()


def get_local_dir_for_trt_llm_engine_hash(trt_llm_hash: str, gpu_model_name: str | None = None) -> pathlib.Path:
    """Get the local directory for a TRT LLM engine hash.

    Args:
        trt_llm_hash: The hash of the TRT LLM engine.
        gpu_model_name: The name of the GPU model.

    Returns:
        The local directory for the TRT LLM engine hash.

    """
    if gpu_model_name is None:
        infos = hardware_info.get_gpu_infos()
        if len(infos) > 1:
            logger.warning("On a computer with multiple GPUs, using GPU 0 as the model name.")
        if len(infos) < 1:
            logger.warning("No gpus found!")
        gpu_model_name = infos[0].name.lower().replace(" ", "_")
    logger.info(f"Using {gpu_model_name} as gpu model name.")
    assert " " not in gpu_model_name
    return environment.CONTAINER_PATHS_ENGINE_CACHE_DIR / gpu_model_name / trt_llm_hash


def _get_cloud_storage_dir_for_weights_name(
    weights_name: str,
    model_weights_prefix: str = environment.MODEL_WEIGHTS_PREFIX,
) -> str:
    """Get the cloud storage directory for a given name for a set of weights.

    Args:
        weights_name: The name of the weights.
        model_weights_prefix: The prefix of the model weights.

    Returns:
        The cloud storage directory for the given name for a set of weights.

    """
    return model_weights_prefix + weights_name + "/"


def get_local_dir_for_weights_name(weights_name: str) -> pathlib.Path:
    """Get the local directory for a given name for a set of weights.

    Args:
        weights_name: The name of the weights.

    Returns:
        The local directory for the given name for a set of weights.

    """
    return environment.CONTAINER_PATHS_MODEL_WEIGHT_CACHE_DIR / weights_name


# TODO: Add this into the storage_utils.
# TODO: This needs to check the size/hash of the files to ensure incomplete files are not being used
def _hack_copydir_to_cloud_storage(
    client: storage_client.StorageClient,
    source: pathlib.Path,
    destination: str,
) -> None:
    """Recursively copies all files from a local directory to a specified location in cloud storage.

    It uses a background uploader for efficiency.

    Args:
        client: The initialized cloud storage client to interact with the cloud storage.
        source: A `pathlib.Path` object representing the local directory to copy from.
        destination: An `str` object representing the destination path in cloud storage.

    """
    # Ensure the source is a directory
    if not source.is_dir():
        error_msg = f"'{source}' is not a directory or does not exist."
        raise ValueError(error_msg)

    paths_and_dests = []
    # Iterate over items in the source directory
    for item in source.rglob("*"):
        relative_path = item.relative_to(source)
        # Create a new path in the destination for this item
        dest_item_path = destination + str(relative_path)

        # If it's a file, copy it
        if item.is_file():
            paths_and_dests.append((item, dest_item_path))

    background_uploader = client.make_background_uploader()
    for path, dest in paths_and_dests:
        background_uploader.add_task_file(path, dest)
    background_uploader.block_until_done()


def _upload_model_weights_to_cloud_storage(
    client: storage_client.StorageClient,
    weights_name: str,
    local_dir: pathlib.Path,
    model_weights_prefix: str = environment.MODEL_WEIGHTS_PREFIX,
) -> str:
    """Upload the model weights from a local directory to cloud storage.

    Args:
        client: An initialized cloud storage client used to interact with cloud storage.
        weights_name: A string representing the name of the model weights.
        local_dir: A `pathlib.Path` object representing the local directory
            where the model weights are stored.
        model_weights_prefix: prefix for model weight

    Returns:
        An `str` object representing the location in cloud storage where the weights are stored.

    """
    if model_weights_prefix == environment.MODEL_WEIGHTS_PREFIX:
        msg = f"{model_weights_prefix=} must be set to a valid S3 prefix."
        raise ValueError(msg)
    storage_dir = _get_cloud_storage_dir_for_weights_name(weights_name, model_weights_prefix)
    logger.info(f"Pushing {weights_name=} to {storage_dir=} ...")
    _hack_copydir_to_cloud_storage(client, local_dir, storage_dir)
    return storage_dir


def _download_model_weights_from_cloud_storage_to_workspace(
    weights_name: str,
    model_weights_prefix: str = environment.MODEL_WEIGHTS_PREFIX,
) -> pathlib.Path:
    """Download model weights from cloud storage to the workspace.

    Args:
        weights_name: Name of the model weights to download.
        model_weights_prefix: Prefix for the model weights in cloud storage.

    Returns:
        Path to the downloaded model weights in the workspace.

    Raises:
        ValueError: If the model weights are not found in cloud storage.

    """

    def value_error(msg: str) -> None:
        raise ValueError(msg)

    destination = get_local_dir_for_weights_name(weights_name)
    if destination.exists():
        logger.info(f"Model weights {weights_name} already exists at {destination}. Skipping ...")
        return destination

    if model_weights_prefix == environment.MODEL_WEIGHTS_PREFIX:
        value_error(f"{model_weights_prefix=} must be set to a valid S3 prefix.")

    destination.mkdir(exist_ok=True, parents=True)
    try:
        storage_dir = _get_cloud_storage_dir_for_weights_name(weights_name, model_weights_prefix)
        logger.info(f"Syncing {weights_name=} from {storage_dir=} to {destination=} ...")
        storage_prefix = storage_utils.path_to_prefix(storage_dir)
        client = storage_utils.get_storage_client(str(storage_prefix))
        if not client:
            value_error(f"Failed to create storage client for {storage_prefix=}")
        client.sync_remote_to_local(  # type: ignore[union-attr]
            storage_prefix,
            destination,
            delete=True,
            chunk_size_bytes=DOWNLOAD_CHUNK_SIZE_BYTES,
        )
        logger.info(f"Done syncing {weights_name=} from {storage_dir=} to {destination=} ...")
    except Exception as e:
        # Incase we are unable to connect to s3, we delete the destination folder and raise the error
        logger.error(
            "Error syncing weights_name=%s from storage_dir=%s to destination=%s: %s",
            weights_name,
            storage_dir,
            destination,
            e,
        )
        destination.rmdir()
        raise

    return destination


def download_model_weights_from_cloud_storage_to_workspace(
    model_weights_names: list[str],
    cloud_storage_model_weights_prefix: str = environment.MODEL_WEIGHTS_PREFIX,
) -> None:
    """Download multiple model weights from cloud storage to the workspace.

    Args:
        model_weights_names: List of model weight names to download.
        cloud_storage_model_weights_prefix: Prefix for the model weights in cloud storage.

    """
    if not model_weights_names:
        return
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _download_model_weights_from_cloud_storage_to_workspace,
                x,
                cloud_storage_model_weights_prefix,
            )
            for x in model_weights_names
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _download_model_weights_from_local_to_workspace(model_weights_name: str, local_model_weights_path: str) -> None:
    """Download model weights from a local path to the workspace.

    Args:
        model_weights_name: Name of the model weights to download.
        local_model_weights_path: Local path where the model weights are stored.

    Raises:
        ValueError: If the model weights are not found in the local path.

    """
    destination = get_local_dir_for_weights_name(model_weights_name)
    if destination.exists():
        logger.info(f"Model weights {model_weights_name} already exists at {destination}. Skipping ...")
        return
    destination.mkdir(exist_ok=True, parents=True)

    # Define the local source directory where the model weights are stored.
    # Adjust the path below as needed to point to the correct local source.
    local_source = pathlib.Path(local_model_weights_path) / model_weights_name

    if not local_source.exists():
        logger.error(
            f"Local source directory for model weights '{model_weights_name}' does not exist at {local_source}.",
        )
        return

    logger.info(f"Copying model weights {model_weights_name} from {local_source} to {destination} ...")
    try:
        shutil.copytree(src=str(local_source), dst=str(destination), dirs_exist_ok=True)
        logger.info(f"Successfully copied model weights {model_weights_name} to {destination}.")
    except Exception as e:  # noqa: BLE001
        logger.error(f"Failed to copy model weights {model_weights_name} to {destination}: {e}")


def download_model_weights_from_local_to_workspace(
    model_weights_names: list[str],
    local_model_weights_path: str,
) -> None:
    """Download multiple model weights from a local path to the workspace.

    Args:
        model_weights_names: List of model weight names to download.
        local_model_weights_path: Local path where the model weights are stored.

    """
    if not model_weights_names:
        return
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _download_model_weights_from_local_to_workspace,
                x,
                local_model_weights_path,
            )
            for x in model_weights_names
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _download_model_weights_from_huggingface_to_workspace(
    model_id: str,
    revision: str | None,
    allow_patterns: list[str] | None,
    destination: pathlib.Path,
    *,
    is_tmp_destination: bool = False,
) -> None:
    """Download model weights from Hugging Face to the workspace.

    Args:
        model_id: Name of the model variant to download.
        revision: Revision of the model to download. If None, the latest revision is used.
        allow_patterns: List of patterns to filter files to download.
        destination: Path where the model weights will be downloaded.
        is_tmp_destination: Whether the destination is temporary.

    Raises:
        ValueError: If the model weights are not found on Hugging Face.

    """
    huggingface_token = None
    cosmos_config = load_config()
    hf = cosmos_config.huggingface
    if hf is not None:
        huggingface_token = hf.api_key
    else:
        logger.warning("huggingface entry not found in config file.")

    if destination.exists() and not is_tmp_destination:
        logger.warning(f"Destination {destination} already exists.")
    logger.info(f"Downloading {model_id} from huggingface to {destination} ...")
    huggingface_hub.snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        token=huggingface_token,
        allow_patterns=allow_patterns,
    )


def _reduce_t5_model_weights(destination: pathlib.Path) -> None:
    """Reduce the size of T5 model weights by removing unnecessary files.

    Args:
        destination: Path to the T5 model weights.

    """
    import collections
    import gc

    import torch

    dst = destination / "pytorch_model.bin.reduced"
    if dst.exists():
        logger.info("Reduced T5 model weight already exists.")
        return

    src = destination / "pytorch_model.bin"
    if not src.exists():
        error_msg = f"Source model file {src} does not exist."
        raise FileNotFoundError(error_msg)

    logger.info("Reducing T5 model weight size...")

    try:
        model = torch.load(src, map_location="cpu", weights_only=False)

        # Extract encoder weights
        encoder = collections.OrderedDict(
            (k, v) for k, v in model.items() if k.startswith("encoder.") or k == "shared.weight"
        )

        # Free up memory immediately
        del model
        gc.collect()

        # Save the filtered encoder weights
        torch.save(encoder, dst)
        logger.info("Successfully saved reduced encoder weights.")

    except Exception as e:
        logger.error(f"Error during reduction: {e}")
        raise
    finally:
        del encoder
        gc.collect()


def download_model_weights_from_huggingface_to_workspace(
    model_id: str,
    revision: str | None,
    allow_patterns: list[str] | None = None,
) -> None:
    """Download model weights from Hugging Face to the workspace.

    Args:
        model_id: Name of the model variant to download.
        revision: Revision of the model to download. If None, the latest revision is used.
        allow_patterns: List of patterns to filter files to download.

    """
    destination = get_local_dir_for_weights_name(model_id)
    _download_model_weights_from_huggingface_to_workspace(model_id, revision, allow_patterns, destination)
    if model_id == "google-t5/t5-11b":
        _reduce_t5_model_weights(destination)


def push_huggingface_model_to_cloud_storage(
    model_id: str,
    revision: str | None,
    allow_patterns: list[str] | None = None,
    client: storage_client.StorageClient | None = None,
    model_weights_prefix: str = environment.MODEL_WEIGHTS_PREFIX,
) -> str:
    """Push a Hugging Face model to cloud storage.

    Args:
        model_id: Name of the model variant to push.
        revision: Revision of the model to push. If None, the latest revision is used.
        allow_patterns: List of patterns to filter files to push.
        client: Cloud storage client to use. If None, a new client will be created.
        model_weights_prefix: Prefix for the model weights in cloud storage.

    Returns:
        The cloud storage path where the model weights were pushed.

    """
    if client is None:
        client = storage_utils.get_storage_client(model_weights_prefix)
        if client is None:
            error_msg = f"Failed to create storage client for {model_weights_prefix=}"
            raise ValueError(error_msg)
    with operation_utils.make_temporary_dir() as tmp_dir:
        _download_model_weights_from_huggingface_to_workspace(
            model_id,
            revision,
            allow_patterns,
            tmp_dir,
            is_tmp_destination=True,
        )
        return _upload_model_weights_to_cloud_storage(client, model_id, tmp_dir, model_weights_prefix)
