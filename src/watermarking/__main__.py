import argparse
from collections.abc import Callable
from pathlib import Path
from typing import cast

import cv2

import watermarking.algorithms.non_blind as NB
from watermarking.attack import Attack
from watermarking.benchmark import Benchmark
from watermarking.plot import plot_ber, plot_psnr
from watermarking.watermark import BlindWatermark, NonBlindWatermark, RGBImage, Watermark

ALGORITHMS = {
    "dct": NB.DCT(),
    "dwt": NB.DWT(),
    "svd": NB.SVD(),
    "dwt-dct-svd": NB.DWT_DCT_SVD(),
}

ATTACKS = {
    "none": Attack.none,
    "noise_gauss": Attack.gaussian_noise,
    "noise_sp": Attack.salt_and_pepper,
    "jpeg": Attack.jpeg_compress,
    "blur": Attack.gaussian_blur,
    "crop": Attack.crop,
    "flip_vertical": Attack.flip_vertical,
    "flip_horizontal": Attack.flip_horizontal,
    "rotate_5": Attack.rotate_5,
    "rotate_30": Attack.rotate_30,
    "rotate_45": Attack.rotate_45,
}

parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-i",
    "--input-image",
    type=Path,
    required=True,
    help="Original image to watermark",
)
_ = parser.add_argument(
    "-w",
    "--watermark-image",
    type=Path,
    required=True,
    help="Watermark image to apply",
)
_ = parser.add_argument(
    "-m",
    "--methods",
    choices=ALGORITHMS.keys(),
    nargs="+",
    default=["dct"],
    help="Watermarking methods to apply",
)
_ = parser.add_argument(
    "-a",
    "--attacks",
    choices=ATTACKS.keys(),
    nargs="+",
    default=["none"],
    help="Attacks to apply",
)
args = parser.parse_args()

selected_methods: dict[str, Watermark] = {name: ALGORITHMS[name] for name in args.methods if name in ALGORITHMS}
selected_attacks: dict[str, Callable[..., RGBImage]] = {name: ATTACKS[name] for name in args.attacks if name in ATTACKS}

input_image_path = Path(args.input_image)
watermark_image_path = Path(args.watermark_image)

# TODO: move image loading into some class method (like RGBImage)
orig_image = cv2.imread(args.input_image)
if orig_image is None:
    raise FileNotFoundError(args.input_image)
orig_image = cast("RGBImage", orig_image)

# watermark has to be grayscale, so we convert it on import
orig_watermark = cv2.imread(args.watermark_image, cv2.IMREAD_GRAYSCALE)
if orig_watermark is None:
    raise FileNotFoundError(args.watermark_image)
_, bw_watermark = cv2.threshold(orig_watermark, 0, 1, cv2.THRESH_BINARY)
orig_watermark = cast("RGBImage", orig_watermark)

results_ber = {}
results_psnr = {}

for method_name, algorithm in selected_methods.items():
    results_ber[method_name] = {}

    watermarked, watermark_shape = algorithm.embed(orig_image, orig_watermark)
    _ = cv2.imwrite(f"{input_image_path.stem}_watermarked_{method_name}.png", watermarked)

    watermark_expected = cv2.resize(bw_watermark * 255, watermark_shape, interpolation=cv2.INTER_LINEAR)
    _ = cv2.imwrite(f"{watermark_image_path.stem}_{method_name}_expected.png", watermark_expected)

    results_psnr[method_name] = Benchmark.psnr(orig_image, watermarked)

    for attack_name, attack in selected_attacks.items():
        attacked = attack(watermarked)
        cv2.imwrite(f"{input_image_path.stem}_{method_name}_att_{attack_name}.png", attacked)

        if isinstance(algorithm, BlindWatermark):
            extracted = algorithm.extract(watermarked, watermark_shape)
        elif isinstance(algorithm, NonBlindWatermark):
            extracted = algorithm.extract(orig_image, watermarked, watermark_shape)
        else:
            msg = "Unknown watermark type"
            raise TypeError(msg)

        _ = cv2.imwrite(f"{watermark_image_path.stem}_ex_{method_name}_{attack_name}.png", extracted)

        ber = Benchmark.ber(watermark_expected, extracted)
        results_ber[method_name][attack_name] = ber


plot_psnr(results_psnr)
plot_ber(results_ber)
