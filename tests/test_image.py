import sys
from pathlib import Path
import pytest
import cv2
import numpy as np
from typing import cast


from watermarking.algorithms.non_blind.dct import DCT
from watermarking.algorithms.non_blind.dwt import DWT
from watermarking.algorithms.non_blind.svd import SVD
from watermarking.algorithms.non_blind.dwt_dct_svd import DWT_DCT_SVD
from watermarking.benchmark import Benchmark


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

@pytest.fixture
def original_image():
    img = cv2.imread(str(ROOT / "tests/tests_data/sample.jpg"))
    assert img is not None
    img = cast("RBGImage", img)
    return img


@pytest.fixture
def watermark():
    wm = cv2.imread(str(ROOT / "tests/tests_data/watermark.png"), cv2.IMREAD_GRAYSCALE)
    assert wm is not None
    _, bw_wm = cv2.threshold(wm, 0, 1, cv2.THRESH_BINARY)
    bw_wm = cast("RBGImage", bw_wm)
    return bw_wm

@pytest.fixture(params=[
    DCT(block_size=8, alpha=10.0, pos=(4,4)),
    DWT(wavelet="haar", alpha=20.0),
    SVD(alpha=10.0),
    DWT_DCT_SVD(wavelet="haar", alpha=20.0)
])
def algorithm(request):
    return request.param


def test_embed_returns_image(algorithm, original_image, watermark):
    watermarked, wm_shape = algorithm.embed(original_image, watermark)

    assert isinstance(watermarked, np.ndarray)
    assert watermarked.shape == original_image.shape
    assert isinstance(wm_shape, tuple)
    assert len(wm_shape) == 2

def test_embed_modifies_image(algorithm, original_image, watermark):
    watermarked, _ = algorithm.embed(original_image, watermark)
    diff = np.mean(np.abs(original_image.astype(np.int16) - watermarked.astype(np.int16)))

    assert diff > 0

def test_extraxt_returns_watermark(algorithm, original_image, watermark):
    watermarked, wm_shape = algorithm.embed(original_image, watermark)
    extracted = algorithm.extract(original_image, watermarked, wm_shape)

    assert isinstance(extracted, np.ndarray)
    assert extracted.ndim == 2


def test_image_quality_psnr(algorithm, original_image, watermark):
    watermarked, _ = algorithm.embed(original_image, watermark)
    psnr = Benchmark.psnr(original_image, watermarked)

    assert psnr > 35

