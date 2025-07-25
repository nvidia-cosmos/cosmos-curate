# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Benchmark tests for the split pipeline using NVCF."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import smart_open  # type: ignore[import-untyped]
from loguru import logger
from rich import print_json

from benchmarks.cloudevent import make_cloudevent, push_cloudevent
from benchmarks.secrets import KratosSecrets, NvcfSecrets, S3Secrets
from benchmarks.summary import make_summary_metrics
from cosmos_curate.client.nvcf_cli.ncf.launcher.nvcf_driver import _get_s3_config_str
from cosmos_curate.client.nvcf_cli.ncf.launcher.nvcf_function import NvcfFunction, NvcfFunctionAlreadyDeployedError


def process_and_report_summary_metrics(  # noqa: PLR0913
    summary_path: str,
    transport_params: dict[str, Any],
    num_nodes: int,
    gpus_per_node: int,
    *,
    caption: bool,
    kratos_metrics_endpoint: str,
    kratos_secrets: KratosSecrets,
) -> None:
    """Process summary data from the pipeline and report metrics.

    Args:
        summary_path: path to summary.json file.
        transport_params: smart_open transport parameters.
        num_nodes: Number of nodes used in the benchmark.
        gpus_per_node: Number of GPUs per node.
        caption: Whether captions are enabled.
        kratos_metrics_endpoint: Endpoint for sending metrics.
        kratos_secrets: Authentication secrets for metrics endpoint.

    """
    logger.info(f"Getting summary metrics from {summary_path}")

    with smart_open.open(summary_path, transport_params=transport_params) as f:
        summary_data = json.load(f)

    summary_metrics = make_summary_metrics(summary_data, num_nodes, gpus_per_node, caption=caption, env="nvcf")

    cloudevent = make_cloudevent(summary_metrics)
    logger.info("Metrics to send to Kratos:")
    print_json(json.dumps(cloudevent, indent=2))

    logger.info(f"Pushing metrics to {kratos_metrics_endpoint}")
    response = push_cloudevent(cloudevent, kratos_metrics_endpoint, kratos_secrets)
    logger.info("Response:")
    print_json(json.dumps(response, indent=2))


def nvcf_split_benchmark(  # noqa: PLR0913
    funcid: str,
    version: str,
    nvcf_secrets: NvcfSecrets,
    s3_secrets: S3Secrets,
    kratos_metrics_token_env: str,
    kratos_bearer_url: str,
    image_repository: str,
    image_tag: str,
    metrics_endpoint: str,
    backend: str,
    gpu: str,
    instance_type: str,
    s3_input_prefix: str,
    s3_output_prefix: str,
    max_concurrency: int,
    limit: int,
    caption: int,
    num_nodes: int,
    gpus_per_node: int,
    kratos_metrics_endpoint: str,
) -> None:
    """Run benchmark tests."""
    nvcf_function = NvcfFunction(
        funcid=funcid,
        version=version,
        key=nvcf_secrets.ngc_key,
        org=nvcf_secrets.ngc_org,
        team="no-team",
    )

    # Load and customize configuration templates
    template_dir = Path(__file__).parent

    with (template_dir / "deploy.json").open() as f:
        deploy_data = json.load(f)

    with (template_dir / "invoke.json").open() as f:
        invoke_data = json.load(f)

    # Update deploy configuration
    deploy_data["configuration"]["image"]["repository"] = image_repository
    deploy_data["configuration"]["image"]["tag"] = image_tag
    deploy_data["configuration"]["metrics"]["remoteWrite"]["endpoint"] = metrics_endpoint

    # Update invoke configuration
    invoke_data["args"].update(
        {
            "input_video_path": s3_input_prefix,
            "output_clip_path": s3_output_prefix,
            "qwen_preprocess_dtype": "uint8" if caption == 1 else "float16",
            "generate_captions": caption == 1,
            "limit": limit,
        }
    )

    # Prepare S3 credentials
    s3_config = f"""[default]
aws_access_key_id = {s3_secrets.aws_access_key_id}
aws_secret_access_key = {s3_secrets.aws_secret_access_key}
aws_region = {s3_secrets.aws_region}
"""

    transport_params = {
        "client_kwargs": {
            "aws_access_key_id": s3_secrets.aws_access_key_id,
            "aws_secret_access_key": s3_secrets.aws_secret_access_key,
            "region_name": s3_secrets.aws_region,
        }
    }
    summary_path = s3_output_prefix + "/summary.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info(f"Benchmarking with {caption=} {num_nodes=}, input: {s3_input_prefix}, output: {s3_output_prefix}")
        tmpdir_path = Path(tmpdir)

        deploy_config = tmpdir_path / "deploy.json"
        invoke_config = tmpdir_path / "invoke.json"
        s3_config_file = tmpdir_path / "s3_cred"

        deploy_config.write_text(json.dumps(deploy_data, indent=2))
        invoke_config.write_text(json.dumps(invoke_data, indent=2))
        s3_config_file.write_text(s3_config)
        s3_config_str = _get_s3_config_str(s3_config_file)

        if s3_config_str is None:
            msg = "Failed to get S3 config string"
            raise ValueError(msg)

        try:
            with nvcf_function.deploy(backend, gpu, instance_type, deploy_config, num_nodes, max_concurrency):
                nvcf_function.invoke(invoke_config, s3_config_str, out_dir=tmpdir_path)

            # Get secrets immediately before reporting - benchmarking time may exceed the token's expiration date.
            kratos_secrets = KratosSecrets.from_env(
                kratos_metrics_token_env,
                kratos_bearer_url,
            )

            process_and_report_summary_metrics(
                summary_path,
                transport_params,
                num_nodes,
                gpus_per_node,
                caption=bool(caption),
                kratos_metrics_endpoint=kratos_metrics_endpoint,
                kratos_secrets=kratos_secrets,
            )

        except NvcfFunctionAlreadyDeployedError:
            logger.error("Function is already deployed, this should not happen, previous benchmark may be running.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark tests on NVCF cluster.")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use.")
    parser.add_argument("--caption", type=int, default=0, help="Whether to use captioning for the benchmark.")
    parser.add_argument("--funcid", type=str, required=True, help="Function ID to run.")
    parser.add_argument("--version", type=str, required=True, help="Function version to use.")
    parser.add_argument(
        "--ngc-org-env",
        type=str,
        required=False,
        default="PERF_NGC_NVCF_ORG",
        help="NGC organization ID environment variable.",
    )
    parser.add_argument(
        "--ngc-key-env",
        type=str,
        required=False,
        default="PERF_NGC_NVCF_API_KEY",
        help="NGC API key environment variable.",
    )
    parser.add_argument(
        "--image-repository", type=str, required=True, help="Image repository to use for the benchmark."
    )
    parser.add_argument("--image-tag", type=str, required=True, help="Image tag to use for the benchmark.")
    parser.add_argument(
        "--metrics-endpoint", type=str, required=True, help="Metrics endpoint to use for the benchmark."
    )
    parser.add_argument("--backend", type=str, required=True, help="Backend to use for the benchmark.")
    parser.add_argument("--gpu", type=str, required=True, help="GPU")
    parser.add_argument("--instance-type", type=str, required=True, help="Instance type..")
    parser.add_argument("--s3-input-prefix", type=str, required=True, help="S3 input prefix.")
    parser.add_argument("--s3-output-prefix", type=str, required=True, help="S3 output prefix.")
    parser.add_argument("--max-concurrency", type=int, required=True, default=2, help="Max concurrency.")
    parser.add_argument(
        "--aws-access-key-id-env",
        type=str,
        required=False,
        default="PERF_AWS_ACCESS_KEY_ID",
        help="AWS access key ID environment variable.",
    )
    parser.add_argument(
        "--aws-secret-access-key-env",
        type=str,
        required=False,
        default="PERF_AWS_SECRET_ACCESS_KEY",
        help="AWS secret access key environment variable.",
    )
    parser.add_argument(
        "--aws-region-env", type=str, required=False, default="PERF_AWS_REGION", help="AWS region environment variable."
    )
    parser.add_argument("--limit", type=int, required=True, default=5000, help="Limit the number of videos to process.")
    parser.add_argument(
        "--kratos-metrics-endpoint",
        type=str,
        required=True,
        help="URL of destination for the metrics to push to Kratos.",
    )
    parser.add_argument(
        "--kratos-bearer-url",
        type=str,
        required=True,
        help="URL of the bearer token endpoint for Kratos.",
    )
    parser.add_argument(
        "--kratos-metrics-token-env",
        type=str,
        required=False,
        default="PERF_KRATOS_METRICS_TOKEN",
        help="Environment variable that contains the token to use to push metrics to Kratos.",
    )
    parser.add_argument("--gpus-per-node", type=int, required=True, default=8, help="Number of GPUs per node.")
    return parser.parse_args()


def main() -> None:
    """Run benchmark tests."""
    args = _parse_args()
    nvcf_secrets = NvcfSecrets.from_env(
        args.ngc_org_env,
        args.ngc_key_env,
    )

    s3_secrets = S3Secrets.from_env(
        args.aws_access_key_id_env,
        args.aws_secret_access_key_env,
        args.aws_region_env,
    )

    nvcf_split_benchmark(
        funcid=args.funcid,
        version=args.version,
        nvcf_secrets=nvcf_secrets,
        s3_secrets=s3_secrets,
        kratos_metrics_token_env=args.kratos_metrics_token_env,
        kratos_bearer_url=args.kratos_bearer_url,
        image_repository=args.image_repository,
        image_tag=args.image_tag,
        metrics_endpoint=args.metrics_endpoint,
        backend=args.backend,
        gpu=args.gpu,
        instance_type=args.instance_type,
        s3_input_prefix=args.s3_input_prefix,
        s3_output_prefix=args.s3_output_prefix,
        max_concurrency=args.max_concurrency,
        limit=args.limit,
        caption=args.caption,
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        kratos_metrics_endpoint=args.kratos_metrics_endpoint,
    )


if __name__ == "__main__":
    main()
