# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test nvcf_split_benchmark."""

from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from benchmarks.secrets import KratosSecrets
from benchmarks.split_pipeline.nvcf_split_benchmark import process_and_report_summary_metrics


@patch("benchmarks.split_pipeline.nvcf_split_benchmark.push_cloudevent")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_cloudevent")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.print_json")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.make_summary_metrics")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.json.load")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.smart_open.open")
@patch("benchmarks.split_pipeline.nvcf_split_benchmark.logger")
def test_process_and_report_summary_metrics_happy_path(  # noqa: PLR0913
    mock_logger: MagicMock,  # noqa: ARG001
    mock_smart_open: MagicMock,
    mock_json_load: MagicMock,
    mock_make_summary_metrics: MagicMock,
    mock_print_json: MagicMock,  # noqa: ARG001
    mock_make_cloudevent: MagicMock,
    mock_push_cloudevent: MagicMock,
) -> None:
    """Test process_and_report_summary_metrics function happy path."""
    # Arrange
    test_summary_path = "s3://bucket/path/summary.json"
    test_transport_params: dict[str, Any] = {
        "client_kwargs": {
            "aws_access_key_id": "test_key",
            "aws_secret_access_key": "test_secret",
            "region_name": "us-east-1",
        }
    }
    test_num_nodes = 2
    test_gpus_per_node = 4
    test_caption = True
    test_kratos_metrics_endpoint = "https://metrics.example.com"
    bearer_token = "test_token"  # noqa: S105
    test_kratos_secrets = KratosSecrets(api_key="test_api", bearer_token=bearer_token)

    # Mock data
    test_summary_data = {"pipeline_run_time": 60, "total_video_duration": 3600}
    test_summary_metrics = {"env": "nvcf", "caption": True, "num_nodes": 2}
    test_cloudevent = {"specversion": "1.0", "data": test_summary_metrics}

    # Configure mocks
    mock_file = mock_open()
    mock_smart_open.return_value = mock_file.return_value
    mock_json_load.return_value = test_summary_data
    mock_make_summary_metrics.return_value = test_summary_metrics
    mock_make_cloudevent.return_value = test_cloudevent
    mock_push_cloudevent.return_value = {"status": "success", "message": "Event pushed successfully"}

    # Act
    process_and_report_summary_metrics(
        summary_path=test_summary_path,
        transport_params=test_transport_params,
        num_nodes=test_num_nodes,
        gpus_per_node=test_gpus_per_node,
        caption=test_caption,
        kratos_metrics_endpoint=test_kratos_metrics_endpoint,
        kratos_secrets=test_kratos_secrets,
    )

    # Assert
    mock_json_load.assert_called_once()
    mock_make_summary_metrics.assert_called_once_with(
        test_summary_data, test_num_nodes, test_gpus_per_node, caption=test_caption, env="nvcf"
    )
    mock_make_cloudevent.assert_called_once_with(test_summary_metrics)
    mock_push_cloudevent.assert_called_once_with(test_cloudevent, test_kratos_metrics_endpoint, test_kratos_secrets)
