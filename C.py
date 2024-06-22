import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032

def blur_image(image, kernel_size=(5, 5), sigma=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    return blurred

original_image = cv2.imread('images/C_original.png')
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
blurred_image = blur_image(gray_image, kernel_size=(15, 15), sigma=0)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_blurred = cv2.calcHist([blurred_image], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(2, 3, 4)
plt.plot(hist_original, color='blue')
plt.title('Original Image Histogram')
plt.subplot(2, 3, 2)
plt.imshow(blurred_image, cmap='gray')
plt.title('Blurred Image')
plt.subplot(2, 3, 5)
plt.plot(hist_blurred, color='blue')
plt.title('Blurred Image Histogram')
modified_image = cv2.imread('images/C_modified.png')
gray_modified = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
blurred_modified = blur_image(gray_modified, kernel_size=(15, 15), sigma=0)
hist_modified = cv2.calcHist([blurred_modified], [0], None, [256], [0, 256])
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
plt.title('Modified Image')
plt.subplot(2, 3, 6)
plt.plot(hist_modified, color='blue')
plt.title('Modified Image Histogram')
plt.tight_layout()
plt.show()
