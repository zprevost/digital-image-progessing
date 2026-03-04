import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------- LOAD ----------
# File paths
rgb_path = r"C:\Users\zoeyp\Downloads\CVSubset\FLAME 3 CV Dataset (Sycan Marsh)\Fire\RGB\Corrected FOV\00592.JPG"
thermal_path = r'C:\Users\zoeyp\Downloads\CVSubset\FLAME 3 CV Dataset (Sycan Marsh)\Fire\Thermal\Raw JPG\00592.JPG'

# Load images
rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE) 

# ---------- NORMALIZATION ----------
def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

# ---------- SPATIAL FILTERS ----------
def gaussian_filter(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

# ---------- FREQUENCY FILTERS ----------
def fft_filters(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows//2 , cols//2

    # Low-pass mask
    mask_low = np.zeros((rows, cols), np.uint8)
    mask_low[crow-30:crow+30, ccol-30:ccol+30] = 1

    # High-pass mask
    mask_high = 1 - mask_low

    low = np.fft.ifft2(np.fft.ifftshift(fshift * mask_low))
    high = np.fft.ifft2(np.fft.ifftshift(fshift * mask_high))

    return np.abs(low), np.abs(high)

# ---------- VISUALIZATION ----------
def show_all(rgb, original, gaussian, bilateral, low, high):
    plt.figure(figsize=(12,8))

    plt.subplot(231); plt.title("RGB"); plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.subplot(232); plt.title("Thermal"); plt.imshow(original, cmap="inferno"); plt.axis("off")
    plt.subplot(233); plt.title("Gaussian"); plt.imshow(gaussian, cmap="inferno"); plt.axis("off")
    plt.subplot(234); plt.title("Bilateral"); plt.imshow(bilateral, cmap="inferno"); plt.axis("off")
    plt.subplot(235); plt.title("Low Pass"); plt.imshow(low, cmap="inferno"); plt.axis("off")
    plt.subplot(236); plt.title("High Pass"); plt.imshow(high, cmap="inferno"); plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---------- MAIN ----------
# Validate loads
if rgb_img is None:
    raise FileNotFoundError(f"RGB image not found: {rgb_path}")

if thermal_img is None:
    raise FileNotFoundError(f"Thermal image not found: {thermal_path}")

# Normalize thermal
thermal_norm = normalize(thermal_img)

# Apply filters
gauss = gaussian_filter(thermal_norm)
bilat = bilateral_filter(thermal_norm)
low, high = fft_filters(thermal_norm)

# Visualize
show_all(rgb_img, thermal_norm, gauss, bilat, low, high)