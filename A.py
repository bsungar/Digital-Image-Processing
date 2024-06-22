import cv2
import numpy as np
import matplotlib.pyplot as plt
#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032

# Negative point processing
def negative_image(image):
    L = image.max()
    neg_image = L - image
    return neg_image

original_path = 'images/A_original.png'
modified_path = 'images/A_modified.png'

img = cv2.imread(original_path, 0)
img_neg = negative_image(img)
hist_original = cv2.calcHist([img],[0],None,[256],[0,256])
hist_negative = cv2.calcHist([img_neg],[0],None,[256],[0,256])

img_modified = cv2.imread(modified_path, 0)
hist_modified = cv2.calcHist([img_modified],[0],None,[256],[0,256])

plt.figure(figsize=(15,15))

plt.subplot(3, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.subplot(3, 2, 2)
plt.plot(hist_original, color='blue')
plt.title('Original Image Histogram')
plt.subplot(3, 2, 3)
plt.imshow(img_neg, cmap='gray')
plt.title('Negative Image')
plt.subplot(3, 2, 4)
plt.plot(hist_negative, color='blue')
plt.title('Negative Image Histogram')
plt.subplot(3, 2, 5)
plt.imshow(img_modified, cmap='gray')
plt.title('Modified Image')

plt.subplot(3, 2, 6)
plt.plot(hist_modified, color='blue')
plt.title('Modified Image Histogram')

plt.tight_layout()
plt.show()
