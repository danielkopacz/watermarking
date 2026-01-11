import cv2
import numpy as np

from watermarking.watermark import RGBImage


class Attack:
    @staticmethod
    def none(image: RGBImage):
        return image

    @staticmethod
    def rotate(image: RGBImage, angle: float) -> RGBImage:
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    @staticmethod
    def rotate_180(image: RGBImage) -> RGBImage:
        return cv2.rotate(image, cv2.ROTATE_180)

    @staticmethod
    def rotate_90(image: RGBImage) -> RGBImage:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    @staticmethod
    def rotate_30(image: RGBImage) -> RGBImage:
        return Attack.rotate(image, -30)

    @staticmethod
    def rotate_45(image: RGBImage) -> RGBImage:
        return Attack.rotate(image, -45)

    @staticmethod
    def rotate_5(image: RGBImage) -> RGBImage:
        return Attack.rotate(image, -5)

    @staticmethod
    def flip_horizontal(image: RGBImage) -> RGBImage:
        return cv2.flip(image, 1)

    @staticmethod
    def flip_vertical(image: RGBImage) -> RGBImage:
        return cv2.flip(image, 0)

    @staticmethod
    def crop(image: RGBImage, percent=0.25) -> RGBImage:
        h, w = image.shape[:2]

        h_cut = int(h * percent)
        w_cut = int(w * percent)

        attacked = image.copy()
        attacked[:h_cut, :w_cut] = 0
        return attacked

    @staticmethod
    def resize(image: RGBImage, scale: float = 0.5) -> RGBImage:
        h, w = image.shape[:2]
        return cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def gaussian_blur(image: RGBImage, kernel_size: int = 5) -> RGBImage:
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def median_blur(image: RGBImage, kernel_size: int = 5) -> RGBImage:
        return cv2.medianBlur(image, kernel_size)

    @staticmethod
    def gaussian_noise(image: RGBImage, sigma: float = 20.0) -> RGBImage:
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        attacked = image.astype(np.float32) + noise
        return attacked.clip(0, 255).astype(np.uint8)

    @staticmethod
    def salt_and_pepper(image: RGBImage, amount: float = 0.01) -> RGBImage:
        out = image.copy()
        h, w = image.shape[:2]
        num = int(amount * h * w)

        coords = np.random.randint(0, h, num), np.random.randint(0, w, num)
        out[coords] = 255

        coords = np.random.randint(0, h, num), np.random.randint(0, w, num)
        out[coords] = 0

        return out

    @staticmethod
    def jpeg_compress(image: RGBImage, quality: int = 50) -> RGBImage:
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, enc = cv2.imencode(".jpg", image, encode_params)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return dec
