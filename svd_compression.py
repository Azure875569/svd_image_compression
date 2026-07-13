import cv2
import numpy as np


# =========================
# Basic settings
# =========================

IMAGE_PATH = "images.jpg"
WINDOW_NAME = "SVD Image Compression"
TRACKBAR_NAME = "k value"


# =========================
# Read the image
# =========================

original_image = cv2.imread(IMAGE_PATH)

if original_image is None:
    raise FileNotFoundError(
        f"Cannot read image: {IMAGE_PATH}\n"
        "Please check the image name and file path."
    )

# Resize the image to prevent the window from becoming too large
# Remove this line if you want to use the original image size
original_image = cv2.resize(original_image, (532, 300))

height, width, channels = original_image.shape
max_k = min(height, width)

print("Image shape:", original_image.shape)
print("Maximum k value:", max_k)


# =========================
# Precompute SVD for each color channel
# =========================

svd_data = []

for channel in range(channels):
    channel_matrix = original_image[:, :, channel].astype(np.float64)

    U, singular_values, VT = np.linalg.svd(
        channel_matrix,
        full_matrices=False
    )

    svd_data.append((U, singular_values, VT))


# =========================
# Reconstruct image using k singular values
# =========================

def reconstruct_image(k):
    reconstructed = np.zeros(
        original_image.shape,
        dtype=np.float64
    )

    for channel in range(channels):
        U, singular_values, VT = svd_data[channel]

        reconstructed_channel = (
            U[:, :k] * singular_values[:k]
        ) @ VT[:k, :]

        reconstructed[:, :, channel] = reconstructed_channel

    reconstructed = np.clip(reconstructed, 0, 255)
    reconstructed = reconstructed.astype(np.uint8)

    return reconstructed


# =========================
# Calculate MSE
# =========================

def calculate_mse(original, compressed):
    original_float = original.astype(np.float64)
    compressed_float = compressed.astype(np.float64)

    mse_value = np.mean(
        (original_float - compressed_float) ** 2
    )

    return mse_value


# =========================
# Calculate PSNR
# =========================

def calculate_psnr(original, compressed):
    mse_value = calculate_mse(original, compressed)

    if mse_value == 0:
        return float("inf")

    psnr_value = 20 * np.log10(
        255.0 / np.sqrt(mse_value)
    )

    return psnr_value


# =========================
# Calculate compression information
# =========================

def calculate_compression(k):
    original_data_size = height * width * channels

    compressed_data_size = (
        k * (height + width + 1) * channels
    )

    compressed_percentage = (
        compressed_data_size / original_data_size
    ) * 100

    compression_ratio = (
        original_data_size / compressed_data_size
    )

    return (
        original_data_size,
        compressed_data_size,
        compressed_percentage,
        compression_ratio
    )


# =========================
# Add labels and information to the display
# =========================

def create_display(
    compressed_image,
    k,
    mse_value,
    psnr_value,
    compressed_percentage,
    compression_ratio
):
    original_display = original_image.copy()
    compressed_display = compressed_image.copy()

    cv2.putText(
        original_display,
        "Original Image",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        compressed_display,
        "Compressed Image",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    comparison = np.hstack(
        (original_display, compressed_display)
    )

    information_height = 170

    canvas = np.zeros(
        (
            comparison.shape[0] + information_height,
            comparison.shape[1],
            3
        ),
        dtype=np.uint8
    )

    canvas[:comparison.shape[0], :] = comparison

    if np.isinf(psnr_value):
        psnr_text = "Infinity"
    else:
        psnr_text = f"{psnr_value:.2f} dB"

    information = [
        f"k value: {k}",
        f"MSE: {mse_value:.4f}",
        f"PSNR: {psnr_text}",
        f"Compressed data size: {compressed_percentage:.2f}%",
        f"Compression ratio: {compression_ratio:.2f} : 1"
    ]

    start_y = comparison.shape[0] + 30
    line_spacing = 30

    for index, text in enumerate(information):
        cv2.putText(
            canvas,
            text,
            (15, start_y + index * line_spacing),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return canvas


# =========================
# Trackbar callback function
# =========================

def update_image(k):
    if k < 1:
        k = 1

        cv2.setTrackbarPos(
            TRACKBAR_NAME,
            WINDOW_NAME,
            k
        )

    compressed_image = reconstruct_image(k)

    mse_value = calculate_mse(
        original_image,
        compressed_image
    )

    psnr_value = calculate_psnr(
        original_image,
        compressed_image
    )

    (
        original_size,
        compressed_size,
        compressed_percentage,
        compression_ratio
    ) = calculate_compression(k)

    display_image = create_display(
        compressed_image,
        k,
        mse_value,
        psnr_value,
        compressed_percentage,
        compression_ratio
    )

    cv2.imshow(
        WINDOW_NAME,
        display_image
    )

    print("-" * 50)
    print("k value:", k)
    print(f"MSE: {mse_value:.4f}")

    if np.isinf(psnr_value):
        print("PSNR: Infinity")
    else:
        print(f"PSNR: {psnr_value:.2f} dB")

    print("Original data size:", original_size)
    print("Compressed data size:", compressed_size)
    print(f"Compressed data percentage: {compressed_percentage:.2f}%")
    print(f"Compression ratio: {compression_ratio:.2f} : 1")


# =========================
# Create window and Trackbar
# =========================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

initial_k = min(50, max_k)

cv2.createTrackbar(
    TRACKBAR_NAME,
    WINDOW_NAME,
    initial_k,
    max_k,
    update_image
)

update_image(initial_k)


# =========================
# Wait for user input
# =========================

print()
print("Instructions:")
print("Move the Trackbar to change the k value.")
print("Press Q or Esc to close the program.")

while True:
    key = cv2.waitKey(30) & 0xFF

    if key == ord("q") or key == 27:
        break

cv2.destroyAllWindows()
