from typing import override

import cv2
import numpy as np

from watermarking.watermark import NonBlindWatermark, RGBImage


class DCT(NonBlindWatermark):
    def __init__(self, block_size=8, alpha=10.0, pos=(4, 4)) -> None:
        self.block_size = block_size
        self.alpha = alpha
        self.pos = pos

    @override
    def embed(self, image: RGBImage, watermark: RGBImage) -> RGBImage:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)

        height, width = Y.shape
        h_blocks, w_blocks = height // self.block_size, width // self.block_size

        _, wm_norm = cv2.threshold(watermark, 0, 1, cv2.THRESH_BINARY)
        wm_res = cv2.resize(wm_norm, (w_blocks, h_blocks), interpolation=cv2.INTER_LINEAR)

        Y_w = np.zeros_like(Y)

        for i in range(h_blocks):
            for j in range(w_blocks):
                block = Y[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ]
                dct_b = cv2.dct(block)

                u, v = self.pos
                dct_b[u, v] += self.alpha * wm_res[i, j]

                Y_w[
                    i * self.block_size : (i + 1) * self.block_size,
                    j * self.block_size : (j + 1) * self.block_size,
                ] = cv2.idct(dct_b)

        ycrcb[:, :, 0] = np.clip(Y_w, 0, 255)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), (w_blocks, h_blocks)

    @override
    def extract(
        self,
        original_image: RGBImage,
        watermarked_image: RGBImage,
        watermark_shape: tuple[int, int],
    ) -> RGBImage:
        y1 = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y2 = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

        h, w = y1.shape
        bs = self.block_size
        H, W = h // bs, w // bs

        extracted = np.zeros((H, W), dtype=np.float32)

        for i in range(H):
            for j in range(W):
                block_orig = y1[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs]
                block_wm = y2[i * bs : (i + 1) * bs, j * bs : (j + 1) * bs]

                dct_o = cv2.dct(block_orig)
                dct_w = cv2.dct(block_wm)

                u, v = self.pos
                extracted[i, j] = (dct_w[u, v] - dct_o[u, v]) / self.alpha

        extracted = np.clip(extracted, 0, 1)
        _, extracted = cv2.threshold(extracted, 0.5, 1.0, cv2.THRESH_BINARY)
        extracted = cv2.resize(extracted, watermark_shape, interpolation=cv2.INTER_LINEAR)
        return (extracted * 255).astype(np.uint8)
