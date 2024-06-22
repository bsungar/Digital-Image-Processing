import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

original_image = cv2.imread('images/D_original.png')
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
sharpened_image = sharpen_image(gray_image)
hist_original = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_sharpened = cv2.calcHist([sharpened_image], [0], None, [256], [0, 256])
plt.figure(figsize=(10, 6))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(2, 3, 4)
plt.plot(hist_original, color='blue')
plt.title('Original Image Histogram')
plt.subplot(2, 3, 2)
plt.imshow(sharpened_image, cmap='gray')
plt.title('Sharpened Image')
plt.subplot(2, 3, 5)
plt.plot(hist_sharpened, color='blue')
plt.title('Sharpened Image Histogram')
modified_image = cv2.imread('images/D_modified.png')
gray_modified = cv2.cvtColor(modified_image, cv2.COLOR_BGR2GRAY)
sharpened_modified = sharpen_image(gray_modified)
hist_modified = cv2.calcHist([sharpened_modified], [0], None, [256], [0, 256])
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(modified_image, cv2.COLOR_BGR2RGB))
plt.title('Modified Image')
plt.subplot(2, 3, 6)
plt.plot(hist_modified, color='blue')
plt.title('Modified Image Histogram')
plt.ylim([0, 7000])
plt.tight_layout()
plt.show()
