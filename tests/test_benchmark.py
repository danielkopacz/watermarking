import pytest
import numpy as np
from watermarking.benchmark import Benchmark


@pytest.fixture
def large_image():
    # Large image 8x8, 3 color channels
    return np.full((8, 8, 3), 100, dtype=np.uint8)

@pytest.fixture
def modified_large_image(large_image):
    img = large_image.copy()
    img[0, 0, 0] += 10
    img[7, 7, 2] -= 10
    img[4, 0, 0] += 10
    return img

@pytest.fixture
def image():
    # Simple image 4x4, 3 color channels
    img = np.array([
        [[10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10]],
        [[50, 50, 50], [50, 50, 50], [50, 50, 50], [50, 50, 50]],
        [[100, 100, 100], [100, 100, 100], [100, 100, 100], [100, 100, 100]],
        [[200, 200, 200], [200, 200, 200], [200, 200, 200], [200, 200, 200]],
    ], dtype=np.uint8)
    return img

@pytest.fixture
def modified_image(image):
    # Changing exactly 3 values ​​by 10 units
    img = image.copy()
    img[0, 0, 0] += 10 
    img[3, 3, 2] -= 10 
    img[2, 0, 0] += 10
    return img


# PSNR tests

def test_psnr_identical(image):
    psnr_value = Benchmark.psnr(image, image)

    assert psnr_value == float("inf")

def test_psnr_modified(image, modified_image):
    psnr_value = Benchmark.psnr(image, modified_image)

    assert psnr_value < float("inf")
    assert psnr_value > 0

# BER tests

def test_ber_identical(image):
    ber_value = Benchmark.ber(image, image)
    
    assert ber_value == 0.0

def test_ber_modified(image, modified_image):
    # 3 modified values out of 48 total -> BER = 0.0625
    ber_value = Benchmark.ber(image, modified_image)

    assert 0 < ber_value <= 1
    assert ber_value == 0.0625

def test_ber_totally_different():
    img1 = np.zeros((10, 10), dtype=np.uint8)
    img2 = np.ones((10, 10), dtype=np.uint8)
    
    assert Benchmark.ber(img1, img2) == 1.0

# SSIM tests

def test_ssim_identical(large_image):
    ssim_value = Benchmark.ssim(large_image, large_image)
    assert ssim_value == 1.0 

def test_ssim_modified(large_image, modified_large_image):
    ssim_value = Benchmark.ssim(large_image, modified_large_image)
    assert 0 < ssim_value < 1 

# NCC tests

def test_ncc_identical(image):
    ncc_value = Benchmark.ncc(image, image)
    assert ncc_value == 1.0 

def test_ncc_partial(image, modified_image):
    ncc_value = Benchmark.ncc(image, modified_image)
    assert 0 < ncc_value < 1 
