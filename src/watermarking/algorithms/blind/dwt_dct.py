from typing import cast, override

import cv2
import numpy as np
import pywt

from watermarking.watermark import BlindWatermark, ImageShape, RGBImage


class DWT_DCT(BlindWatermark):
    def __init__(self, wavelet: str = "haar", qim_step: int = 30, block_size: int = 4):
        self.wavelet: str = wavelet
        self.qim_step: int = qim_step
        self.block_size: int = block_size

    def _qim_embed(self, coefficient: float, bit: int):
        if bit == 0:  # quantize to even multiple of step
            return round(coefficient / self.qim_step) * self.qim_step
        else:  # quantize to odd multiple of step
            return round((coefficient - (self.qim_step / 2)) / self.qim_step) * self.qim_step + (self.qim_step / 2)

    def _qim_extract(self, coefficient: float):
        val_0 = round(coefficient / self.qim_step) * self.qim_step
        val_1 = round((coefficient - (self.qim_step / 2)) / self.qim_step) * self.qim_step + (self.qim_step / 2)

        dist_0 = abs(coefficient - val_0)
        dist_1 = abs(coefficient - val_1)

        return 0 if dist_0 < dist_1 else 1

    @override
    def embed(self, image: RGBImage, watermark: RGBImage):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image_ycrcb[:, :, 0].astype(np.float32)
        watermark_bin = self._image_to_binary(watermark)

        # extract DWT coefficients
        coeffs = pywt.dwt2(y_channel, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        height, width = LL.shape

        # crop the dimensions to fit the blocks
        height = height - (height % self.block_size)
        width = width - (width % self.block_size)

        # check capacity
        max_bits = (height // self.block_size) * (width // self.block_size)
        watermark_length = len(watermark_bin)

        if watermark_length > max_bits:
            msg = f"Capacity exceeded: max bits: {max_bits}, watermark length: {watermark_length}"
            raise ValueError(msg)

        watermark_bit_index = 0

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                if watermark_bit_index >= watermark_length:
                    break

                block = LL[i : i + self.block_size, j : j + self.block_size]
                dct_block = cv2.dct(block)  # pyright: ignore[reportCallIssue, reportArgumentType]

                coeff = dct_block[0, 1]
                dct_block[0, 1] = self._qim_embed(coeff, watermark_bin[watermark_bit_index])

                LL[i : i + self.block_size, j : j + self.block_size] = cv2.idct(dct_block)
                watermark_bit_index += 1

        y_channel_watermarked = pywt.idwt2((LL, (LH, HL, HH)), self.wavelet)

        # Handle DWT padding odd image dimensions
        original_height, original_width = y_channel.shape
        y_channel_watermarked = y_channel_watermarked[:original_height, :original_width]

        image_ycrcb[:, :, 0] = np.clip(y_channel_watermarked, 0, 255).astype(np.uint8)
        watermarked_img = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_img = cast("RGBImage", watermarked_img)

        return watermarked_img, watermark.shape

    @override
    def extract(self, watermarked_image: RGBImage, watermark_shape: ImageShape):
        image_ycrcb = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)
        y_channel = image_ycrcb[:, :, 0].astype(np.float32)

        coeffs = pywt.dwt2(y_channel, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        height, width = LL.shape
        height = height - (height % self.block_size)
        width = width - (width % self.block_size)

        extracted_bits = []
        total_bits = np.prod(watermark_shape)
        watermark_bit_index = 0

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                if watermark_bit_index >= total_bits:
                    break

                block = LL[i : i + self.block_size, j : j + self.block_size]
                dct_block = cv2.dct(block)  # pyright: ignore[reportCallIssue, reportArgumentType]

                coeff = dct_block[0, 1]
                bit = self._qim_extract(coeff)

                extracted_bits.append(bit)
                watermark_bit_index += 1

        extracted_bits = np.array(extracted_bits, dtype=np.uint8) * 255
        return extracted_bits.reshape(watermark_shape)
