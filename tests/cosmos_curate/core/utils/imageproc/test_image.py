"""Tests for cosmos_curate.core.utils.imageproc.image."""

import io
import pathlib

import numpy as np
import PIL.features
import PIL.Image
import pytest
from PIL import UnidentifiedImageError

from cosmos_curate.core.utils.imageproc.image import Image


def _make_image_bytes(size: tuple[int, int], color: tuple[int, int, int], format_name: str) -> bytes:
    image = PIL.Image.new("RGB", size=size, color=color)
    with io.BytesIO() as buffer:
        image.save(buffer, format=format_name)
        return buffer.getvalue()


def _make_image_bytes_with_mode(*, size: tuple[int, int], mode: str, color: tuple[int, ...], format_name: str) -> bytes:
    image = PIL.Image.new(mode, size=size, color=color)
    with io.BytesIO() as buffer:
        image.save(buffer, format=format_name)
        return buffer.getvalue()


def test_from_numpy_array_sets_metadata_dimensions() -> None:
    """Image creation from a numpy array sets metadata dimensions."""
    array = np.zeros((5, 8, 3), dtype=np.uint8)

    image = Image.from_numpy_array(array)

    assert image.metadata.dimensions.width == 8
    assert image.metadata.dimensions.height == 5
    np.testing.assert_array_equal(image.array, array)


def test_from_jpeg_preserves_shape_and_values() -> None:
    """Loading from JPEG preserves reported size and pixel values (within tolerance)."""
    jpeg_bytes = _make_image_bytes(size=(4, 3), color=(12, 34, 56), format_name="JPEG")

    image = Image.from_jpeg(jpeg_bytes)

    assert image.metadata.dimensions.width == 4
    assert image.metadata.dimensions.height == 3
    expected = np.full((3, 4, 3), fill_value=[12, 34, 56], dtype=np.uint8)
    np.testing.assert_allclose(image.array, expected, atol=2)


def test_to_jpeg_produces_jpeg_magic_bytes() -> None:
    """Serializing to JPEG produces expected magic bytes."""
    array = np.random.default_rng(0).integers(0, 256, size=(6, 7, 3), dtype=np.uint8)
    image = Image.from_numpy_array(array)

    jpeg_bytes = image.to_jpeg()

    assert jpeg_bytes.startswith(b"\xff\xd8")
    assert jpeg_bytes.endswith(b"\xff\xd9")


@pytest.mark.parametrize(
    ("image_type", "format_name"),
    [
        ("jpg", "JPEG"),
        ("png", "PNG"),
        ("gif", "GIF"),
        pytest.param(
            "webp",
            "WEBP",
            marks=pytest.mark.skipif(
                not PIL.features.check("webp"),  # type: ignore[attr-defined]
                reason="Pillow was built without WEBP support",
            ),
        ),
    ],
)
def test_from_jpeg_or_gif_or_webp_or_png(image_type: str, format_name: str) -> None:
    """Supported image types are decoded into an Image with matching metadata."""
    raw_bytes = _make_image_bytes(size=(5, 4), color=(200, 100, 50), format_name=format_name)

    image = Image.from_jpeg_or_gif_or_webp_or_png(raw_bytes, image_type=image_type)

    assert image.metadata.dimensions.width == 5
    assert image.metadata.dimensions.height == 4
    assert image.array.shape == (4, 5, 3)


def test_from_jpeg_or_gif_or_webp_or_png_with_unknown_type() -> None:
    """Unsupported image types raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported image type"):
        Image.from_jpeg_or_gif_or_webp_or_png(b"", image_type="bmp")


def test_from_png_preserves_shape_and_values() -> None:
    """PNG loader preserves dimensions and RGB payload."""
    png_bytes = _make_image_bytes(size=(7, 2), color=(10, 20, 30), format_name="PNG")

    image = Image.from_png(png_bytes)

    assert image.metadata.dimensions.width == 7
    assert image.metadata.dimensions.height == 2
    expected = np.full((2, 7, 3), fill_value=[10, 20, 30], dtype=np.uint8)
    np.testing.assert_array_equal(image.array, expected)


def test_to_jpeg_file_writes_expected_payload(tmp_path: pathlib.Path) -> None:
    """JPEG file helper writes bytes identical to to_jpeg()."""
    array = np.random.default_rng(1).integers(0, 256, size=(3, 4, 3), dtype=np.uint8)
    image = Image.from_numpy_array(array)
    expected_bytes = image.to_jpeg()
    out_path = tmp_path / "image.jpg"

    image.to_jpeg_file(out_path)

    assert out_path.read_bytes() == expected_bytes


def test_from_jpeg_round_trip_preserves_dimensions() -> None:
    """Saving to JPEG and reloading yields the same metadata."""
    array = np.random.default_rng(2).integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
    image = Image.from_numpy_array(array)

    round_trip = Image.from_jpeg(image.to_jpeg())

    assert round_trip.metadata.dimensions == image.metadata.dimensions


def test_is_mostly_black_threshold_behavior() -> None:
    """Mostly-black heuristic returns True for black frames and False for bright ones."""
    black_array = np.zeros((10, 10, 3), dtype=np.uint8)
    bright_array = np.full((10, 10, 3), fill_value=255, dtype=np.uint8)

    assert Image.from_numpy_array(black_array).is_mostly_black() is True
    assert Image.from_numpy_array(bright_array).is_mostly_black() is False


def test_is_mostly_black_threshold_boundary() -> None:
    """Boundary around the 0.1 threshold behaves as expected."""
    below_threshold = np.full((4, 4, 3), fill_value=20, dtype=np.uint8)
    above_threshold = np.full((4, 4, 3), fill_value=40, dtype=np.uint8)

    assert Image.from_numpy_array(below_threshold).is_mostly_black() is True
    assert Image.from_numpy_array(above_threshold).is_mostly_black() is False


def test_from_numpy_array_handles_extreme_aspect_ratios() -> None:
    """Skinny or tall arrays are accepted and metadata is correct."""
    wide = np.zeros((1, 50, 3), dtype=np.uint8)
    tall = np.zeros((60, 2, 3), dtype=np.uint8)

    wide_image = Image.from_numpy_array(wide)
    tall_image = Image.from_numpy_array(tall)

    assert wide_image.metadata.dimensions.width == 50
    assert wide_image.metadata.dimensions.height == 1
    assert tall_image.metadata.dimensions.width == 2
    assert tall_image.metadata.dimensions.height == 60


def test_loaders_convert_non_rgb_color_spaces() -> None:
    """Images in grayscale or RGBA colour spaces convert to RGB arrays."""
    grayscale_bytes = _make_image_bytes_with_mode(size=(3, 2), mode="L", color=(25,), format_name="PNG")
    rgba_bytes = _make_image_bytes_with_mode(size=(3, 2), mode="RGBA", color=(100, 150, 200, 123), format_name="PNG")

    grayscale_image = Image.from_png(grayscale_bytes)
    rgba_image = Image.from_png(rgba_bytes)

    assert grayscale_image.array.shape == (2, 3, 3)
    assert rgba_image.array.shape == (2, 3, 3)
    expected_gray = np.full((2, 3, 3), 25, dtype=np.uint8)
    expected_rgba = np.tile(np.array([[[100, 150, 200]]], dtype=np.uint8), (2, 3, 1))
    np.testing.assert_array_equal(grayscale_image.array, expected_gray)
    np.testing.assert_array_equal(rgba_image.array, expected_rgba)


def test_from_jpeg_raises_with_corrupt_bytes() -> None:
    """Invalid JPEG data surfaces an image identification error."""
    with pytest.raises(UnidentifiedImageError):
        Image.from_jpeg(b"not-a-jpeg")
