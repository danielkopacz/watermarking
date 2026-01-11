from typing import override

import cv2
import numpy as np

from watermarking.watermark import NonBlindWatermark, RGBImage


class SVD(NonBlindWatermark):
    def __init__(self, alpha=10.0) -> None:
        self.alpha = alpha

    @override
    def embed(self, image: RGBImage, watermark: RGBImage) -> RGBImage:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)

        h, w = Y.shape

        U, S, Vt = np.linalg.svd(Y, full_matrices=True)

        k = min(h, w)

        _, wm_norm = cv2.threshold(watermark, 0, 1, cv2.THRESH_BINARY)
        wm_res = cv2.resize(wm_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        Sigma = np.zeros((h, w), dtype=np.float32)
        Sigma[:k, :k] = np.diag(S)

        Sigma_w = Sigma + (self.alpha * wm_res)

        Y_w = U @ Sigma_w @ Vt

        ycrcb[:, :, 0] = np.clip(Y_w, 0, 255)

        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), (k, k)

    @override
    def extract(self, original_image: RGBImage, watermarked_image: RGBImage, watermark_shape) -> RGBImage:
        y_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y_wm = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

        U, S, Vt = np.linalg.svd(y_orig, full_matrices=True)
        h, w = y_orig.shape

        Sigma_rec = U.T @ y_wm @ Vt.T
        Sigma_orig = np.zeros((h, w), dtype=np.float32)
        k = min(h, w)
        Sigma_orig[:k, :k] = np.diag(S)

        extracted = (Sigma_rec - Sigma_orig) / self.alpha

        extracted = np.clip(extracted, 0, 1)
        _, extracted = cv2.threshold(extracted, 0.5, 1.0, cv2.THRESH_BINARY)
        extracted = cv2.resize(extracted, watermark_shape, interpolation=cv2.INTER_LINEAR)

        return (extracted * 255).astype(np.uint8)
