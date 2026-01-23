from typing import cast, override

import cv2
import numpy as np
import pywt

from watermarking.watermark import ImageShape, NonBlindWatermark, RGBImage


class DWT_DCT_SVD(NonBlindWatermark):
    def __init__(self, wavelet="haar", alpha=20.0) -> None:
        self.wavelet = wavelet
        self.alpha = alpha

    @override
    def embed(self, image: RGBImage, watermark: RGBImage) -> tuple[RGBImage, ImageShape]:
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        Y = ycrcb[:, :, 0].astype(np.float32)
        h_orig, w_orig = Y.shape

        coeffs = pywt.dwt2(Y, self.wavelet)
        LL, (LH, HL, HH) = coeffs

        h_ll, w_ll = LL.shape
        pad_h = h_ll % 2
        pad_w = w_ll % 2

        if pad_h or pad_w:
            LL = cv2.copyMakeBorder(LL, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)  # pyright: ignore[reportCallIssue, reportArgumentType, reportConstantRedefinition]

        dct_ll = cv2.dct(LL)  # pyright: ignore[reportCallIssue, reportArgumentType]

        U, s, Vt = np.linalg.svd(dct_ll, full_matrices=False)

        k = len(s)
        Sigma = np.diag(s)

        _, wm_norm = cv2.threshold(watermark, 0, 1, cv2.THRESH_BINARY)
        wm_res = cv2.resize(wm_norm, (k, k), interpolation=cv2.INTER_LINEAR)
        wm_float = wm_res.astype(np.float32)

        Sigma_w = Sigma + (self.alpha * wm_float)

        dct_w = U @ Sigma_w @ Vt

        LL_w_padded = cv2.idct(dct_w)

        LL_w = LL_w_padded[:h_ll, :w_ll] if pad_h or pad_w else LL_w_padded

        coeffs_w = (LL_w, (LH, HL, HH))
        Y_w = pywt.idwt2(coeffs_w, self.wavelet)

        Y_w = Y_w[:h_orig, :w_orig]
        Y_w = np.clip(Y_w, 0, 255)

        ycrcb[:, :, 0] = Y_w
        watermarked_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        watermarked_image = cast("RGBImage", watermarked_image)
        return watermarked_image, (k, k)

    @override
    def extract(self, original_image: RGBImage, watermarked_image: RGBImage, watermark_shape) -> RGBImage:
        y_orig = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)
        y_wm = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32)

        LL_orig, _ = pywt.dwt2(y_orig, self.wavelet)
        LL_wm, _ = pywt.dwt2(y_wm, self.wavelet)

        h_ll, w_ll = LL_orig.shape
        pad_h = h_ll % 2
        pad_w = w_ll % 2

        if pad_h or pad_w:
            LL_orig = cv2.copyMakeBorder(LL_orig, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)  # pyright: ignore[reportCallIssue, reportArgumentType]
            LL_wm = cv2.copyMakeBorder(LL_wm, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)  # pyright: ignore[reportCallIssue, reportArgumentType]

        dct_orig = cv2.dct(LL_orig)  # pyright: ignore[reportCallIssue, reportArgumentType]
        dct_wm = cv2.dct(LL_wm)  # pyright: ignore[reportCallIssue, reportArgumentType]

        U_orig, s_orig, Vt_orig = np.linalg.svd(dct_orig, full_matrices=False)
        Sigma_orig = np.diag(s_orig)

        Sigma_rec = U_orig.T @ dct_wm @ Vt_orig.T

        extracted = (Sigma_rec - Sigma_orig) / self.alpha
        extracted = (extracted > 0.5).astype(np.float32)

        extracted = cv2.resize(extracted, watermark_shape, interpolation=cv2.INTER_NEAREST)

        return (extracted * 255).astype(np.uint8)
