# Watermarking

## Installation

Clone the repository
```bash
git clone https://github.com/danielkopacz/watermarking.git
```

Create and activate virtual environment
```bash
cd watermarking
python -m venv .venv

# Linux
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1
```

Install dependencies
```bash
pip install -e .
```

## Running

Image watermarking example
```bash
python -m watermarking -i sample.jpg -w watermark.png -m dct_nb -a none noise_gauss rotate_30
```

Video watermarking example
```bash
python -m video_watermarking -i sample.mp4 -w watermark.png -m dct_nb -o watermarked.mp4
```

## Available watermarking methods

| Method  | Description |
| ------------- | ------------- |
| dct_nb  | Discrete Cosine Transform (DCT)  |
| dwt_nb  | Discrete Wavelet Transform (DWT)  |
| svd_nb  | Singular Value Decomposition (SVD)  |
| dwt-dct-svd_nb  | Hybrid DWT-DCT-SVD  |
| dwt-dct-svd  | Hybrid DWT-DCT-SVD  |
| dwt-dct  | Hybrid DWT-DCT  |


## Available attacks

| Attack  | Description |
| ------------- | ------------- |
| none  | No attack applied  |
| noise_gauss  | Add Gaussian noise  |
| noise_sp  | Add salt-and-pepper noise  |
| jpeg  | JPEG compression  |
| blur  | Image blurring  |
| cutout  | Cutout of image  |
| flip_vertical  | Vertical flip  |
| flip_horizontal  | Horizontal flip  |
| rotate_5  | Rotate image by 5°  |
| rotate_30  | Rotate image by 30°  |
| rotate_45  | Rotate image by 45°  |


