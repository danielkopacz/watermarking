import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast, override

import cv2

import watermarking.algorithms.blind as B
import watermarking.algorithms.non_blind as NB
from watermarking.watermark import BlindWatermark, NonBlindWatermark, RGBImage, Watermark

logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(name)s] %(levelname)s: %(message)s", level=logging.DEBUG)


@dataclass
class VideoInfo:
    # path: str
    frame_count: int
    fps: float
    width: int
    height: int
    codec_fourcc: int
    pixel_format_fourcc: int

    @override
    def __repr__(self) -> str:
        return f"Frame count: {self.frame_count}, FPS: {self.fps}, Width: {self.width}, Height: {self.height}, Codec: {self.codec}, Pixel format: {self.pixel_format}"

    @staticmethod
    def _get_fourcc(fourcc: float) -> str:
        return int(fourcc).to_bytes(4, byteorder=sys.byteorder).decode()

    @property
    def codec(self) -> str:
        return self._get_fourcc(self.codec_fourcc)

    @property
    def pixel_format(self) -> str:
        return self._get_fourcc(self.pixel_format_fourcc)


def get_video_info(video: cv2.VideoCapture):
    return VideoInfo(
        frame_count=int(video.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=video.get(cv2.CAP_PROP_FPS),
        width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        codec_fourcc=int(video.get(cv2.CAP_PROP_FOURCC)),
        pixel_format_fourcc=int(video.get(cv2.CAP_PROP_CODEC_PIXEL_FORMAT)),
    )


def video_to_frames(video_path):
    video_capture = cv2.VideoCapture(video_path)
    video_info = get_video_info(video_capture)
    frames = []
    for frame in range(video_info.frame_count):
        success, image = video_capture.read()
        if not success:
            logger.warning(f"Error processing frame {frame}")
            break
        frames.append(image)
    video_capture.release()
    return frames, video_info


def frames_to_video(frames, video_info: VideoInfo, output_path: str) -> None:
    video_capture = cv2.VideoWriter(
        filename=output_path,
        # fourcc=video_info.codec_fourcc,
        fourcc=cv2.VideoWriter.fourcc(*"mp4v"),
        fps=video_info.fps,
        frameSize=(video_info.width, video_info.height),
    )
    for frame in frames:
        video_capture.write(frame)
    video_capture.release()


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


ALGORITHMS = {
    "dct": NB.DCT(),
    "dwt": NB.DWT(),
    "svd": NB.SVD(),
    "dwt-dct-svd": NB.DWT_DCT_SVD(),
    "dwt-dct": B.DWT_DCT(qim_step=50),
}

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
