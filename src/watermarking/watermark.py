from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

type RGBImage = NDArray[np.uint8]


class Watermark(ABC):
    @abstractmethod
    def embed(self, image: RGBImage, watermark: RGBImage) -> RGBImage:
        """Embed the watermark into the image."""


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
