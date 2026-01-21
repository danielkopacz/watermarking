import cv2
import numpy as np
import pywt

from watermarking.watermark import BlindWatermark


class DWT_DCT(BlindWatermark):
    def __init__(self, qim_step: int = 30, block_size: int = 4):
        self.qim_step: int = qim_step
        self.block_size: int = block_size

    def _qim_embed(self, c1, c2, bit):
        step = self.qim_step
        difference = c1 - c2

        if bit == 0:  # quantize diff to even multiple of step
            target_diff = round(difference / step) * step
        else:  # quantize diff to odd multiple of step
            target_diff = round((difference - (step / 2)) / step) * step + (step / 2)

        shift_needed = target_diff - difference

        # apply half the shift to c1 and subtract half from c2 to keep the average energy the same
        c1_new = c1 + (shift_needed / 2)
        c2_new = c2 - (shift_needed / 2)

        return c1_new, c2_new

    def _qim_extract(self, c1, c2):
        step = self.qim_step
        difference = c1 - c2

        val_0 = round(difference / step) * step
        val_1 = round((difference - (step / 2)) / step) * step + (step / 2)

        dist_0 = abs(difference - val_0)
        dist_1 = abs(difference - val_1)

        return 0 if dist_0 < dist_1 else 1

    def embed(self, image, watermark):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image_ycrcb[:, :, 0].astype(np.float32)
        watermark_bin = self._image_to_binary(watermark)

        # extract DWT coefficients
        coeffs = pywt.dwt2(y_channel, "haar")
        LL, (LH, HL, HH) = coeffs

        # crop the dimensions to fit the blocks
        original_height, original_width = LL.shape

        height = original_height - (original_height % self.block_size)
        width = original_width - (original_width % self.block_size)

        # check capacity
        max_bits = (height // self.block_size) * (width // self.block_size)
        watermark_length = len(watermark_bin)

        if watermark_length > max_bits:
            watermark_bin = watermark_bin[:max_bits]
            print(f"Capacity exceeded: max bits: {max_bits}, watermark length: {watermark_length}")

        watermark_bit_index = 0

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                if watermark_bit_index >= watermark_length:
                    break

                block = LL[i : i + self.block_size, j : j + self.block_size]
                dct_block = cv2.dct(block)

                # coefficients (0,1) and (1,0)
                c1 = dct_block[0, 1]
                c2 = dct_block[1, 0]

                c1_new, c2_new = self._qim_embed(c1, c2, watermark_bin[watermark_bit_index])

                dct_block[0, 1] = c1_new
                dct_block[1, 0] = c2_new

                idct_block = cv2.idct(dct_block)
                LL[i : i + self.block_size, j : j + self.block_size] = idct_block
                watermark_bit_index += 1

        coeffs_new = (LL, (LH, HL, HH))
        y_channel_watermarked = pywt.idwt2(coeffs_new, "haar")

        image_ycrcb[:, :, 0] = np.clip(y_channel_watermarked, 0, 255).astype(np.uint8)
        watermarked_img = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)

        return watermarked_img, watermark.shape

    def extract(self, image, watermark_shape):
        image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = image_ycrcb[:, :, 0].astype(np.float32)

        coeffs = pywt.dwt2(y_channel, "haar")
        LL, _ = coeffs

        original_height, original_width = LL.shape
        height = original_height - (original_height % self.block_size)
        width = original_width - (original_width % self.block_size)

        extracted_bits = []
        total_bits = np.prod(watermark_shape)
        watermark_bit_index = 0

        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                if watermark_bit_index >= total_bits:
                    break

                block = LL[i : i + self.block_size, j : j + self.block_size]
                dct_block = cv2.dct(block)

                c1 = dct_block[0, 1]
                c2 = dct_block[1, 0]
                bit = self._qim_extract(c1, c2)

                extracted_bits.append(bit)
                watermark_bit_index += 1

        extracted_bits = np.array(extracted_bits, dtype=np.uint8) * 255
        return extracted_bits.reshape(watermark_shape)
