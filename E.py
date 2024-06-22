import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032

def median_filter(image, kernel_size):
    filtered_image = cv2.medianBlur(image, kernel_size)
    return filtered_image


original_image = cv2.imread('images/E_original.png', 0)
modified_image = cv2.imread('images/E_modified.png', 0)


kernel_size = 11  
filtered_original_image = median_filter(original_image, kernel_size)
filtered_modified_image = median_filter(modified_image, kernel_size)

hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])
hist_modified = cv2.calcHist([modified_image], [0], None, [256], [0, 256])
hist_filtered_original = cv2.calcHist([filtered_original_image], [0], None, [256], [0, 256])

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(original_image, cmap='gray')
plt.title('Original Image')

plt.subplot(3, 2, 2)
plt.plot(hist_original, color='blue')
plt.title('Original Image Histogram')

plt.subplot(3, 2, 3)
plt.imshow(modified_image, cmap='gray')
plt.title('Modified Image')

plt.subplot(3, 2, 4)
plt.plot(hist_modified, color='blue')
plt.title('Modified Image Histogram')

plt.subplot(3, 2, 5)
plt.imshow(filtered_original_image, cmap='gray')
plt.title('Filtered Original Image')

plt.subplot(3, 2, 6)
plt.plot(hist_filtered_original, color='blue')
plt.title('Filtered Original Image Histogram')

plt.tight_layout()
plt.show()
