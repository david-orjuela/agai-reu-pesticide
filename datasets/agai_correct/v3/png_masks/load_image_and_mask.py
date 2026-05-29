import cv2
import numpy as np
import matplotlib.pyplot as plt

# Paths
image_path = r"C:\Users\david\OneDrive - University of Central Florida\AgAI_REU_2026\Datasets\AgAI Correct\V3 - 2026-03-25 1415\PNG Masks\train\ICP_600ppm_4_jpg.rf.7e7383a82fd37a49e34669e30683eff3.jpg"
mask_path = r"C:\Users\david\OneDrive - University of Central Florida\AgAI_REU_2026\Datasets\AgAI Correct\V3 - 2026-03-25 1415\PNG Masks\train\ICP_600ppm_4_jpg.rf.7e7383a82fd37a49e34669e30683eff3_mask.png"

# Load image (BGR -> RGB)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load mask (grayscale)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Normalize mask to binary (0 or 1)
mask_binary = (mask > 0).astype(np.uint8)

# Create colored mask (red overlay)
mask_colored = np.zeros_like(image)
mask_colored[:, :, 0] = mask_binary * 255  # Red channel

# Overlay mask on image
alpha = 0.5
overlay = cv2.addWeighted(image, 1, mask_colored, alpha, 0)

# Plot everything
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Mask")
plt.imshow(mask_binary, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(overlay)
plt.axis("off")

plt.tight_layout()
plt.show()