# SVD Image Compression

An interactive image compression demo using Singular Value Decomposition (SVD), built with OpenCV and NumPy.

## Demo

Adjust the **k value** slider to control the number of singular values used for reconstruction. A lower k means higher compression; a higher k means better image quality.

The window displays the original and compressed images side by side, along with real-time metrics:
- **MSE** (Mean Squared Error)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **Compression ratio**

## How It Works

Each color channel (B, G, R) of the image is decomposed using SVD:

$$A = U \Sigma V^T$$

The image is then approximated using only the top **k** singular values:

$$A_k = U_{:,1:k} \cdot \Sigma_{1:k} \cdot V^T_{1:k,:}$$

A smaller k retains less information but requires less storage. The compressed data size is calculated as:

$$k \times (H + W + 1) \times C$$

where H, W, C are the image height, width, and number of channels.

## Requirements

```
opencv-python
numpy
```

Install with:

```bash
pip install opencv-python numpy
```

## Usage

1. Place your image in the same directory and rename it to `images.jpg`
2. Run the script:

```bash
python svd_compression.py
```

3. Move the trackbar to adjust the k value
4. Press **Q** or **Esc** to exit

## Controls

| Key | Action |
|-----|--------|
| Trackbar | Adjust k value (1 ~ min(H, W)) |
| Q / Esc | Close the window |
