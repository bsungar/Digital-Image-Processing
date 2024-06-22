import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032

original_image = cv2.imread('images/B_original.png', 0)

if original_image is None:
    print("Error: Unable to load original image.")
else:
    # Perform histogram equalization
    equalized_image = cv2.equalizeHist(original_image)

    hist_original = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(2, 3, 4)
    plt.plot(hist_original, color='blue')
    plt.title('Original Image Histogram')
    plt.subplot(2, 3, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')

    plt.subplot(2, 3, 5)
    plt.plot(hist_equalized, color='blue')
    plt.title('Equalized Image Histogram')
    modified_image = cv2.imread('images/B_modified.png', 0)

    if modified_image is None:
        print("Error: Unable to load modified image.")
    else:
        
        equalized_modified = cv2.equalizeHist(modified_image)
        hist_modified = cv2.calcHist([modified_image], [0], None, [256], [0, 256])
        hist_equalized_modified = cv2.calcHist([equalized_modified], [0], None, [256], [0, 256])
        plt.subplot(2, 3, 3)
        plt.imshow(modified_image, cmap='gray')
        plt.title('Modified Image')

        plt.subplot(2, 3, 6)
        plt.plot(hist_equalized_modified, color='blue')
        plt.title('Modified Image Histogram')

    plt.tight_layout()
    plt.show()
