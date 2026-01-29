import argparse
from collections.abc import Callable
from pathlib import Path

import cv2

import watermarking.algorithms.blind as Blind
import watermarking.algorithms.non_blind as NonBlind
from watermarking.attack import Attack
from watermarking.benchmark import Benchmark
from watermarking.plot import plot_ber, plot_ncc, plot_psnr, plot_ssim, plot_jpeg, plot_ng
from watermarking.watermark import BlindWatermark, NonBlindWatermark, RGBImage, Watermark

ALGORITHMS = {
    "dct_nb": NonBlind.DCT(),
    "dwt_nb": NonBlind.DWT(),
    "svd_nb": NonBlind.SVD(),
    "dwt-dct-svd_nb": NonBlind.DWT_DCT_SVD(alpha=10),
    "dwt-dct-svd": Blind.DWT_DCT_SVD(qim_step=50),
    "dwt-dct": Blind.DWT_DCT(qim_step=50),
}

ATTACKS = {
    "none": Attack.none,
    "noise_gauss": Attack.gaussian_noise,
    "noise_sp": Attack.salt_and_pepper,
    "jpeg": Attack.jpeg_compress,
    "blur": Attack.gaussian_blur,
    "cutout": Attack.cutout,
    "flip_vertical": Attack.flip_vertical,
    "flip_horizontal": Attack.flip_horizontal,
    "rotate_5": Attack.rotate_5,
    "rotate_30": Attack.rotate_30,
    "rotate_45": Attack.rotate_45,
}

JPEG_QUALITIES = list(range(0, 101, 10)) 
GAUSSIAN_SIGMAS = list(range(0, 51, 5))

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

image_path = Path(args.input_image)
watermark_path = Path(args.watermark_image)

image = Watermark.load_image(image_path)
watermark = Watermark.load_image(watermark_path, cv2.IMREAD_GRAYSCALE)  # import as grayscale
_, watermark_bw = cv2.threshold(watermark, 0, 1, cv2.THRESH_BINARY)

results_ber = {}
results_psnr = {}
results_ssim = {}
results_ncc = {}
if "jpeg" in selected_attacks:
    results_ber_jpeg = {}
    results_ncc_jpeg = {}
if "noise_gauss" in selected_attacks:
    results_ber_ng = {}
    results_ncc_ng = {}

for method_name, algorithm in selected_methods.items():
    results_ber[method_name] = {}
    results_ncc[method_name] = {}
    if "jpeg" in selected_attacks:
        results_ber_jpeg[method_name] = {}
        results_ncc_jpeg[method_name] = {}
    if "noise_gauss" in selected_attacks:
        results_ber_ng[method_name] = {}
        results_ncc_ng[method_name] = {}
    watermarked, watermark_shape = algorithm.embed(image, watermark)
    _ = cv2.imwrite(f"{image_path.stem}_watermarked_{method_name}.png", watermarked)

    # We embed binary watermark bits, so for visual comparison and benchmarking prepare an "expected watermark" image
    expected = cv2.resize(watermark_bw * 255, watermark_shape, interpolation=cv2.INTER_LINEAR)
    _ = cv2.imwrite(f"{watermark_path.stem}_{method_name}_expected.png", expected)

    results_psnr[method_name] = Benchmark.psnr(image, watermarked)
    results_ssim[method_name] = Benchmark.ssim(image, watermarked)

    for attack_name, attack in selected_attacks.items():
        if "noise_gauss" in selected_attacks:
            results_ber_ng[method_name]["noise_gauss"] = []
            results_ncc_ng[method_name]["noise_gauss"] = []

            for sigma in GAUSSIAN_SIGMAS:
                attacked = Attack.gaussian_noise(watermarked, sigma=sigma)
                cv2.imwrite(f"{image_path.stem}_{method_name}_att_gauss_sigma{sigma}.png", attacked)

                if isinstance(algorithm, BlindWatermark):
                    extracted = algorithm.extract(attacked, watermark_shape)
                elif isinstance(algorithm, NonBlindWatermark):
                    extracted = algorithm.extract(image, attacked, watermark_shape)
                else:
                    raise TypeError("Unknown watermark type")

                cv2.imwrite(f"{watermark_path.stem}_ex_{method_name}_gauss_sigma{sigma}.png", extracted)

                ber = Benchmark.ber(expected, extracted)
                results_ber_ng[method_name]["noise_gauss"].append((sigma, ber))

                ncc = Benchmark.ncc(expected, extracted)
                results_ncc_ng[method_name]["noise_gauss"].append((sigma, ncc))
        elif attack_name == 'jpeg':
            results_ber_jpeg[method_name]["jpeg"] = []
            results_ncc_jpeg[method_name]["jpeg"] = []

            for q in JPEG_QUALITIES:
                attacked = Attack.jpeg_compress(watermarked, quality=q)
                cv2.imwrite(f"{image_path.stem}_{method_name}_att_jpeg_q{q}.png", attacked)

                if isinstance(algorithm, BlindWatermark):
                    extracted = algorithm.extract(attacked, watermark_shape)
                elif isinstance(algorithm, NonBlindWatermark):
                    extracted = algorithm.extract(image, attacked, watermark_shape)
                else:
                    raise TypeError("Unknown watermark type")

                _ = cv2.imwrite(f"{watermark_path.stem}_ex_{method_name}_jpeg_q{q}.png", extracted)

                ber = Benchmark.ber(expected, extracted)
                results_ber_jpeg[method_name]["jpeg"].append((q, ber))

                ncc = Benchmark.ncc(expected, extracted)
                results_ncc_jpeg[method_name]["jpeg"].append((q, ncc))
        else:
            attacked = attack(watermarked)
            cv2.imwrite(f"{image_path.stem}_{method_name}_att_{attack_name}.png", attacked)

            if isinstance(algorithm, BlindWatermark):
                extracted = algorithm.extract(attacked, watermark_shape)
            elif isinstance(algorithm, NonBlindWatermark):
                extracted = algorithm.extract(image, attacked, watermark_shape)
            else:
                raise TypeError("Unknown watermark type")

            _ = cv2.imwrite(f"{watermark_path.stem}_ex_{method_name}_{attack_name}.png", extracted)

            ber = Benchmark.ber(expected, extracted)
            results_ber[method_name][attack_name] = ber

            ncc = Benchmark.ncc(expected, extracted)
            results_ncc[method_name][attack_name] = ncc


plot_psnr(results_psnr)
plot_ber(results_ber)
plot_ssim(results_ssim)
plot_ncc(results_ncc)
if "jpeg" in selected_attacks:
    plot_jpeg(results_ber_jpeg, "BER")
    plot_jpeg(results_ncc_jpeg, "NCC")
if "noise_gauss" in selected_attacks:
    plot_ng(results_ber_ng, "BER")
    plot_ng(results_ncc_ng, "NCC")