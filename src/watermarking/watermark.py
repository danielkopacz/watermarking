from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

type RGBImage = NDArray[np.uint8]


class Watermark(ABC):
    @abstractmethod
    def embed(self, image: RGBImage, watermark: RGBImage) -> RGBImage:
        """Embed the watermark into the image."""

    @staticmethod
    def load_image(image_path: Path, flags: int | None = cv2.IMREAD_COLOR_BGR) -> RGBImage:
        image = cv2.imread(image_path, flags)
        if image is None:
            raise FileNotFoundError(image_path)
        return image

    @staticmethod
    def _image_to_binary(image: RGBImage):
        _, image_bin = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)
        return image_bin.flatten()


class BlindWatermark(Watermark, ABC):
    @abstractmethod
    def extract(self, watermarked_image: RGBImage, watermark_shape: tuple[int, int]) -> RGBImage:
        """Extract the watermark from the watermarked image."""


class NonBlindWatermark(Watermark, ABC):
    @abstractmethod
    def extract(
        self,
        original_image: RGBImage,
        watermarked_image: RGBImage,
        watermark_shape: tuple[int, int],
    ) -> RGBImage:
        """Extract the watermark from the watermarked image."""
