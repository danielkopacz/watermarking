import numpy as np
from skimage.metrics import structural_similarity

from watermarking.watermark import RGBImage


class Benchmark:
    @staticmethod
    def psnr(original: RGBImage, watermarked: RGBImage) -> float:
        original = original.astype(np.float32)
        watermarked = watermarked.astype(np.float32)

        mse = np.mean((original - watermarked) ** 2)
        if mse == 0:
            return float("inf")

        max_val = 255.0
        return 20 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def ber(watermark_expected, extracted_watermark):
        return np.mean(watermark_expected != extracted_watermark)

    @staticmethod
    def ssim(original: RGBImage, watermarked: RGBImage):
        return structural_similarity(original, watermarked, channel_axis=2, data_range=255)

    @staticmethod
    def ncc(watermark_expected, extracted_watermark):
        v1 = watermark_expected.flatten().astype(np.float32)
        v2 = extracted_watermark.flatten().astype(np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)
