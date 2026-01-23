import logging
import sys
from dataclasses import dataclass
from typing import override

import cv2

logger = logging.getLogger(__name__)


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
