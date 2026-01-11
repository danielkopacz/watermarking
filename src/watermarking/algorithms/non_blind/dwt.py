from typing import override

import cv2
import numpy as np
import pywt

from watermarking.watermark import NonBlindWatermark, RGBImage


class DWT(NonBlindWatermark):
    def __init__(self, wavelet="haar", alpha=20.0) -> None:
        self.wavelet = wavelet
        self.alpha = alpha

    @override
    def embed(self, image: RGBImage, watermark: RGBImage) -> RGBImage:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)

        coeffs = pywt.wavedec2(Y, self.wavelet, level=2)
        LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs

        target_subband = LH1

        h_sb, w_sb = target_subband.shape

        _, wm_norm = cv2.threshold(watermark, 0, 1, cv2.THRESH_BINARY)
        wm_res = cv2.resize(wm_norm, (w_sb, h_sb), interpolation=cv2.INTER_LINEAR)

        target_subband += self.alpha * wm_res
        coeffs_new = [LL2, (LH2, HL2, HH2), (LH1, HL1, HH1)]

        Y_w = pywt.waverec2(coeffs_new, self.wavelet)
        Y_w = Y_w[: Y.shape[0], : Y.shape[1]]

        ycrcb[:, :, 0] = np.clip(Y_w, 0, 255)
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), (w_sb, h_sb)

    @override
    def extract(self, original_image: RGBImage, watermarked_image: RGBImage, watermark_shape) -> RGBImage:
        y_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y_wm = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

        coeffs_orig = pywt.wavedec2(y_orig, self.wavelet, level=2)
        coeffs_wm = pywt.wavedec2(y_wm, self.wavelet, level=2)

        LL2_orig, (LH2_orig, HL2_orig, HH2_orig), (LH1_orig, HL1_orig, HH1_orig) = coeffs_orig
        LL2_wm, (LH2_wm, HL2_wm, HH2_wm), (LH1_wm, HL1_wm, HH1_wm) = coeffs_wm

        sb_orig = LH1_orig
        sb_wm = LH1_wm

        extracted = (sb_wm - sb_orig) / self.alpha

        extracted = np.clip(extracted, 0, 1)
        _, extracted = cv2.threshold(extracted, 0.5, 1.0, cv2.THRESH_BINARY)

        extracted = cv2.resize(extracted, watermark_shape, interpolation=cv2.INTER_LINEAR)
        return (extracted * 255).astype(np.uint8)
