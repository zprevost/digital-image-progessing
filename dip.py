import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ---------- LOAD ----------
# File paths
rgb_path = r"C:\Users\zoeyp\Downloads\CVSubset\FLAME 3 CV Dataset (Sycan Marsh)\Fire\RGB\Corrected FOV\00592.JPG"
thermal_path = r"C:\Users\zoeyp\Downloads\CVSubset\FLAME 3 CV Dataset (Sycan Marsh)\Fire\Thermal\Raw JPG\00592.JPG"

rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
thermal_img = cv2.imread(thermal_path, cv2.IMREAD_GRAYSCALE)

if rgb_img is None:
    raise FileNotFoundError(f"RGB image not found: {rgb_path}")
if thermal_img is None:
    raise FileNotFoundError(f"Thermal image not found: {thermal_path}")

print("Images loaded successfully.")
print("RGB Shape:", rgb_img.shape)
print("Thermal Shape:", thermal_img.shape)

# ---------- NORMALIZATION ----------
def normalize(img):
    img = img.astype(np.float32)
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

thermal_norm = normalize(thermal_img)

# ---------- SPATIAL FILTERS ----------
# Gaussian Filter
# Kernel = 5x5 window
# Sigma = 0 (auto-computed from kernel size)
gaussian = cv2.GaussianBlur(thermal_norm, (5,5), 0)

# Bilateral Filter
# d = 9 (neighborhood diameter)
# sigmaColor = 75 (intensity similarity)
# sigmaSpace = 75 (spatial distance)
bilateral = cv2.bilateralFilter(
    (thermal_norm*255).astype(np.uint8),
    d=9,
    sigmaColor=75,
    sigmaSpace=75
)

# Median Filter (Impulse noise removal)
median = cv2.medianBlur(
    (thermal_norm*255).astype(np.uint8),
    5
)

# Non-Local Means (Advanced denoising)
nlm = cv2.fastNlMeansDenoising(
    (thermal_norm*255).astype(np.uint8),
    None,
    h=10,
    templateWindowSize=7,
    searchWindowSize=21
)

# ---------- FREQUENCY FILTERS ----------
def fft_filters(img):

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2

    # Low-pass mask (retain center frequencies)
    mask_low = np.zeros((rows, cols))
    mask_low[crow-30:crow+30, ccol-30:ccol+30] = 1

    # High-pass mask
    mask_high = 1 - mask_low

    low_freq = np.fft.ifft2(np.fft.ifftshift(fshift * mask_low))
    high_freq = np.fft.ifft2(np.fft.ifftshift(fshift * mask_high))

    return np.abs(low_freq), np.abs(high_freq)

low_pass, high_pass = fft_filters(thermal_norm)

# -------- Gaussian Pyramid --------
gp = [ (thermal_norm*255).astype(np.uint8) ]
G = gp[0]

for i in range(3):
    G = cv2.pyrDown(G)
    gp.append(G)

# -------- Wavelet Transform --------
coeffs2 = pywt.dwt2(thermal_norm, 'haar')
LL, (LH, HL, HH) = coeffs2

# ---------- VISUALIZATION ----------
# ---- Spatial Comparison ----
plt.figure(figsize=(12,8))
plt.subplot(231); plt.title("RGB"); plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.subplot(232); plt.title("Original Thermal"); plt.imshow(thermal_norm, cmap="inferno"); plt.axis("off")
plt.subplot(233); plt.title("Gaussian"); plt.imshow(gaussian, cmap="inferno"); plt.axis("off")
plt.subplot(234); plt.title("Median"); plt.imshow(median, cmap="inferno"); plt.axis("off")
plt.subplot(235); plt.title("Bilateral"); plt.imshow(bilateral, cmap="inferno"); plt.axis("off")
plt.subplot(236); plt.title("Non-Local Means"); plt.imshow(nlm, cmap="inferno"); plt.axis("off")
plt.tight_layout()
plt.show()

# ---- Frequency Domain ----
plt.figure(figsize=(10,4))
plt.subplot(121); plt.title("Low Pass"); plt.imshow(low_pass, cmap="inferno"); plt.axis("off")
plt.subplot(122); plt.title("High Pass"); plt.imshow(high_pass, cmap="inferno"); plt.axis("off")
plt.tight_layout()
plt.show()

# ---- Gaussian Pyramid ----
plt.figure(figsize=(8,6))
for i in range(len(gp)):
    plt.subplot(2,2,i+1)
    plt.title(f"Pyramid Level {i}")
    plt.imshow(gp[i], cmap="inferno")
    plt.axis("off")
plt.tight_layout()
plt.show()

# ---- Wavelet Components ----
plt.figure(figsize=(8,6))
plt.subplot(221); plt.title("Approximation (LL)"); plt.imshow(LL, cmap="inferno"); plt.axis("off")
plt.subplot(222); plt.title("Horizontal Detail (LH)"); plt.imshow(LH, cmap="inferno"); plt.axis("off")
plt.subplot(223); plt.title("Vertical Detail (HL)"); plt.imshow(HL, cmap="inferno"); plt.axis("off")
plt.subplot(224); plt.title("Diagonal Detail (HH)"); plt.imshow(HH, cmap="inferno"); plt.axis("off")
plt.tight_layout()
plt.show()

print("All tasks (1–5) executed successfully.")
