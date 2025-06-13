# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Test annotation_writer_stage module."""

try:
    import pathlib
    import uuid
    from collections.abc import Generator
    from unittest.mock import MagicMock, patch
    from uuid import UUID

    import pytest

    from cosmos_curate.core.utils.s3_client import S3Client, S3Prefix
    from cosmos_curate.pipelines.av.utils.av_data_model import (
        AvClipAnnotationTask,
    )
    from cosmos_curate.pipelines.av.writers.annotation_writer_stage import (
        AnnotationJsonWriterStage,
        _annotation_json_writer,
        _get_json_annotation_url,
    )
    from tests.cosmos_curate.pipelines.av.writers.test_utils import (
        create_mock_clip,
        create_test_annotation_task,
        run_writer_stage_test,
    )
except ImportError:
    pass


@pytest.mark.parametrize(
    ("output_prefix", "expected_type"),
    [
        ("s3://bucket/path", S3Prefix),
        ("/local/path", pathlib.Path),
    ],
)
def test_get_json_annotation_url_return_type(output_prefix: str, expected_type: type) -> None:
    """Test that _get_json_annotation_url returns the correct type."""
    with patch(
        "cosmos_curate.core.utils.s3_client.is_s3path",
        return_value=output_prefix.startswith("s3://"),
    ):
        result = _get_json_annotation_url(
            prefix=output_prefix,
            clip_uuid=UUID("12345678-1234-5678-1234-567812345678"),
        )
        assert isinstance(result, expected_type)


def test_annotation_json_writer() -> None:
    """Test that _annotation_json_writer correctly processes clips and writes JSON."""
    # Setup
    mock_s3_client = MagicMock()
    output_prefix = "/test/path"
    env = "test_env"
    verbose = True
    overwrite = False

    # Create a mock clip
    clip_uuid = uuid.uuid4()
    mock_clip = create_mock_clip(clip_uuid=clip_uuid)

    task = create_test_annotation_task(single_clip=True)

    # Mock the _get_json_annotation_url function
    expected_output_url = pathlib.Path(f"{output_prefix}/{env}/annotation/v9000/{clip_uuid}_{mock_clip.camera_id}.json")

    with (
        patch(
            "cosmos_curate.pipelines.av.writers.annotation_writer_stage._get_json_annotation_url",
            return_value=expected_output_url,
        ) as mock_get_url,
        patch("cosmos_curate.pipelines.av.writers.annotation_writer_stage.write_json") as mock_write_json,
    ):
        # Call the function
        _annotation_json_writer(
            s3_client=mock_s3_client,
            task=task,
            output_prefix=output_prefix,
            overwrite=overwrite,
            verbose=verbose,
        )

        # Verify _get_json_annotation_url was called with correct parameters
        mock_get_url.assert_called_once_with(
            output_prefix,
            task.clips[0].uuid,
        )

        _EXPECTED_CALL_COUNT = 2
        mock_write_json.assert_called()
        assert mock_write_json.call_count == _EXPECTED_CALL_COUNT


@pytest.mark.parametrize(
    "output_prefix",
    [
        "/test/path",  # Local path
        "s3://bucket/test/path",  # S3 path
    ],
)
def test_annotation_json_writer_stage(output_prefix: str) -> None:
    """Test the AnnotationJsonWriterStage class happy path with both local and S3 paths.

    Args:
        output_prefix: The output prefix to test (local or S3 path)

    """
    # Define test parameters
    tasks = [create_test_annotation_task()]

    # Define expected writer arguments
    def _get_writer_args(
        stage: AnnotationJsonWriterStage,
        tasks: list[AvClipAnnotationTask],
    ) -> Generator[tuple[S3Client | None, AvClipAnnotationTask, str, bool, bool], None, None]:
        for task in tasks:
            yield (
                stage._s3_client,  # noqa: SLF001
                task,
                stage._output_prefix,  # noqa: SLF001
                stage._verbose,  # noqa: SLF001
                stage._overwrite,  # noqa: SLF001
            )

    stage = AnnotationJsonWriterStage(
        output_prefix=output_prefix,
        log_stats=True,
    )
    # Test the stage
    run_writer_stage_test(
        stage=stage,
        tasks=tasks,
        writer_function_path="cosmos_curate.pipelines.av.writers.annotation_writer_stage._annotation_json_writer",
    )
