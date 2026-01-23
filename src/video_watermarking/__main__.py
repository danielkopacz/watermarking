import argparse
import logging
from pathlib import Path

import cv2

import watermarking.algorithms.blind as Blind
import watermarking.algorithms.non_blind as NonBlind
from video_watermarking.video import frames_to_video, video_to_frames
from watermarking.watermark import BlindWatermark, NonBlindWatermark, Watermark

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s] %(levelname)s: %(message)s", level=logging.DEBUG)

ALGORITHMS = {
    "dct_nb": NonBlind.DCT(),
    "dwt_nb": NonBlind.DWT(),
    "svd_nb": NonBlind.SVD(),
    "dwt-dct-svd_nb": NonBlind.DWT_DCT_SVD(alpha=10),
    "dwt-dct-svd": Blind.DWT_DCT_SVD(qim_step=50),
    "dwt-dct": Blind.DWT_DCT(qim_step=50),
}


def embed(input_file: str, output_file: str, watermark, algorithm):
    frames, info = video_to_frames(input_file)

    for i in range(len(frames)):
        frames[i], shape = algorithm.embed(frames[i], watermark)

    frames_to_video(frames, info, output_file)
    return shape


def extract(original_video: str, watermarked_video: str, algorithm, shape: tuple[int, int]):
    orig_frames, _ = video_to_frames(original_video)
    frames, _ = video_to_frames(watermarked_video)

    Path("out").mkdir(parents=True, exist_ok=True)
    for i in range(len(frames)):
        if isinstance(algorithm, BlindWatermark):
            extracted = algorithm.extract(frames[i], shape)
        elif isinstance(algorithm, NonBlindWatermark):
            extracted = algorithm.extract(orig_frames[i], frames[i], shape)
        else:
            msg = "Unknown watermark type"
            raise TypeError(msg)

        cv2.imwrite(f"out/out-{i}.png", extracted)


parser = argparse.ArgumentParser()
_ = parser.add_argument(
    "-i",
    "--input",
    type=Path,
    required=True,
    help="Video to watermark",
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
    "--method",
    choices=ALGORITHMS.keys(),
    required=True,
    default=["dct"],
    help="Watermarking method to apply",
)
_ = parser.add_argument(
    "-o",
    "--output",
    type=Path,
    required=True,
    help="Watermarked video file",
)
args = parser.parse_args()

selected_method = ALGORITHMS.get(args.method)

video_path = Path(args.input)
watermark_path = Path(args.watermark_image)
watermark = Watermark.load_image(watermark_path, cv2.IMREAD_GRAYSCALE)

shape = embed(args.input, args.output, watermark, selected_method)
extract(args.input, args.output, selected_method, shape)
