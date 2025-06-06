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

"""Provide utility functions to handle and manipulate image data."""

import io
import pathlib

import attrs
import cv2
import numpy as np
import numpy.typing as npt
import PIL.Image
from skimage import color

from cosmos_curate.core.utils.dataset_utils import dimensions

# We work with huge images sometimes. PIL gives a lot of false positives where it warns that the images are too large.
# We add this line to ignore those errors.
PIL.Image.MAX_IMAGE_PIXELS = None
# We have massive images sometimes. We need to get PIL to accept them.
PIL.Image.MAX_IMAGE_PIXELS = 933120000

_MOSTLY_BLACK_THRESHOLD = 0.1


@attrs.define
class Metadata:
    """A class representing the metadata associated with an image.

    Currently, this metadata only contains the image's dimensions.
    """

    dimensions: dimensions.Dimensions


@attrs.define
class Image:
    """Represents an image with its pixel data and metadata.

    Attributes:
        array (npt.NDArray): The raw pixel data of the image.
        metadata (Metadata): The metadata associated with the image.

    """

    array: npt.NDArray  # type: ignore[type-arg]
    metadata: Metadata

    @classmethod
    def from_numpy_array(cls, array: npt.NDArray) -> "Image":  # type: ignore[type-arg]
        """Create an Image object from a given numpy array.

        Args:
            array (npt.NDArray): The raw pixel data.

        Returns:
            Image: The created Image object.

        """
        h, w = array.shape[:2]  # Extract height and width
        return Image(array, Metadata(dimensions.Dimensions(w, h)))

    @classmethod
    def from_jpeg(cls, data: bytes) -> "Image":
        """Create an Image object from given JPEG data.

        Args:
            data (bytes): The raw JPEG data.

        Returns:
            Image: The created Image object.

        """
        with io.BytesIO(data) as f:
            image = PIL.Image.open(f, formats=["jpeg"])
            image.load()
            array = np.array(image.convert("RGB"))
        return cls.from_numpy_array(array)

    @classmethod
    def from_jpeg_or_gif_or_webp_or_png(cls, data: bytes, image_type: str) -> "Image":
        """Create an Image object from given JPEG or GIF or WEBP data.

        Args:
            data (bytes): The raw JPEG data.
            image_type (str): jpeg or gif or webp

        Returns:
            Image: The created Image object.

        """
        with io.BytesIO(data) as f:
            if image_type == "jpg":
                image = PIL.Image.open(f, formats=["jpeg"])
            elif image_type == "gif":
                image = PIL.Image.open(f, formats=["GIF"])
            elif image_type == "webp":
                image = PIL.Image.open(f, formats=["WEBP"])
            elif image_type == "png":
                image = PIL.Image.open(f, formats=["PNG"])
            else:
                error_msg = f"Unsupported image type: {image_type}"
                raise ValueError(error_msg)
            image.load()
            array = np.array(image.convert("RGB"))
        return cls.from_numpy_array(array)

    @classmethod
    def from_png(cls, data: bytes) -> "Image":
        """Create an Image object from given JPEG data.

        Args:
            data (bytes): The raw JPEG data.

        Returns:
            Image: The created Image object.

        """
        with io.BytesIO(data) as f:
            image = PIL.Image.open(f, formats=["png"])
            image.load()
            array = np.array(image.convert("RGB"))
        return cls.from_numpy_array(array)

    def to_jpeg(self) -> bytes:
        """Convert the Image object to JPEG format.

        Returns:
            bytes: The converted JPEG data.

        """
        pil_image = PIL.Image.fromarray(self.array)
        with io.BytesIO() as output:
            pil_image.save(output, format="JPEG")
            return output.getvalue()

    def to_jpeg_file(self, path: pathlib.Path) -> None:
        """Convert the Image object to JPEG format and saves it to a file.

        Returns:
            None

        """
        with path.open("wb") as f:
            f.write(self.to_jpeg())

    def is_mostly_black(self) -> bool:
        """Determine if the image is mostly black.

        An image is considered mostly black if both its mean and standard deviation are below a certain threshold.

        Returns:
            bool: True if the image is mostly black, False otherwise.

        """
        # Convert the image to grayscale
        image = color.rgb2gray(self.array)

        # Calculate mean and standard deviation of pixel values
        mean, stddev = cv2.meanStdDev(image)
        mean, stddev = float(mean), float(stddev)
        mostly_black: bool = mean < _MOSTLY_BLACK_THRESHOLD and stddev < _MOSTLY_BLACK_THRESHOLD
        return mostly_black
