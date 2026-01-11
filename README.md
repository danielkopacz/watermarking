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
python -m watermarking -i sample.jpg -w watermark.png -m dct -a none noise_gauss rotate_30
```

Video watermarking example
```bash
python -m video_watermarking -i sample.mp4 -w watermark.png -m dct -o watermarked.mp4
```
