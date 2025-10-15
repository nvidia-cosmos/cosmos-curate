"""Tests for the image caption schema helpers."""

import pickle

import cattrs
import pytest

from cosmos_curate.core.utils.dataset.dimensions import Dimensions
from cosmos_curate.core.utils.dataset.webdataset_utils import (
    RawSample,
    make_tar_from_samples,
    read_raw_samples_from_archive,
)
from cosmos_curate.core.utils.imageproc.image_captions import Captions, Debug, Sample


@pytest.fixture
def sample() -> Sample:
    """Create a canonical Sample instance for tests."""
    return Sample(
        key="sample-key",
        image_resolution=Dimensions(width=640, height=480),
        had_parse_issue=False,
        captions=Captions(kosmos_2="caption-a", llava="caption-b", vfc="caption-c"),
        debug=Debug(gemma_prompt="prompt", llava_num_tokens_in=123, llava_num_tokens_out=456),
    )


def test_sample_to_dict_and_from_dict_roundtrip(sample: Sample) -> None:
    """Structured dict round-trips through Sample conversion helpers."""
    as_dict = sample.to_dict()
    rebuilt = Sample.from_dict(as_dict)

    assert rebuilt == sample
    assert as_dict["captions"]["kosmos_2"] == "caption-a"
    assert as_dict["image_resolution"]["width"] == 640
    assert as_dict["debug"]["llava_num_tokens_out"] == 456


def test_sample_to_webdataset_sample_contains_pickled_payload(sample: Sample) -> None:
    """Samples can be exported to RawSample with a pickled payload."""
    raw_sample = sample.to_webdataset_sample()

    assert isinstance(raw_sample, RawSample)
    assert raw_sample.key == "sample-key"
    assert set(raw_sample.data) == {"pkl"}

    payload = pickle.loads(raw_sample.data["pkl"])  # noqa: S301 - trusted fixture data for tests
    assert payload == sample.to_dict()


def test_sample_from_dict_missing_field_raises_class_validation_error(sample: Sample) -> None:
    """Missing required fields surface a cattrs validation error."""
    data = sample.to_dict()
    del data["captions"]

    with pytest.raises(cattrs.errors.ClassValidationError):
        Sample.from_dict(data)


def test_sample_tar_round_trip_restores_original_sample(sample: Sample) -> None:
    """Sample survives Sample→RawSample→tar→RawSample→Sample round-trip."""
    raw_sample = sample.to_webdataset_sample()

    tar_bytes = make_tar_from_samples([raw_sample])
    extracted = read_raw_samples_from_archive(tar_bytes, archive_type="tar")

    assert len(extracted) == 1
    assert extracted[0].key == sample.key
    rebuilt_dict = pickle.loads(extracted[0].data["pkl"])  # noqa: S301 - trusted fixture data for tests
    rebuilt_sample = Sample.from_dict(rebuilt_dict)

    assert rebuilt_sample == sample
