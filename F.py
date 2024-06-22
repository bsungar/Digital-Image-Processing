import cv2
import numpy as np
import matplotlib.pyplot as plt

#Mustafa Eren Tuğcu 200316035
#Beyza Sungar 200316032

def gamma_correction(image, gamma=1.0):
    gamma_corrected = np.power(image / 255.0, gamma) * 255
    gamma_corrected = np.uint8(gamma_corrected)
    return gamma_corrected

image = cv2.imread('images/F_original.png', 0)

if image is None:
    print("Error: Unable to load image.")
else:
    gamma = 0.5
    gamma_corrected_image = gamma_correction(image, gamma)

    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_gamma_corrected = cv2.calcHist([gamma_corrected_image], [0], None, [256], [0, 256])

    plt.figure(figsize=(10, 9))
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(3, 2, 2)
    plt.imshow(gamma_corrected_image, cmap='gray')
    plt.title('Gamma Corrected Image')
    plt.subplot(3, 2, 3)
    plt.plot(hist_original, color='blue')
    plt.title('Original Image Histogram')
    plt.subplot(3, 2, 4)
    plt.plot(hist_gamma_corrected, color='blue')
    plt.title('Gamma Corrected Image Histogram')
    modified_image = cv2.imread('images/F_modified.png', 0)

    if modified_image is None:
        print("Error: Unable to load modified image.")
    else:
        gamma_corrected_modified = gamma_correction(modified_image, gamma)

        hist_modified = cv2.calcHist([modified_image], [0], None, [256], [0, 256])
        hist_gamma_corrected_modified = cv2.calcHist([gamma_corrected_modified], [0], None, [256], [0, 256])

        plt.subplot(3, 2, 5)
        plt.imshow(modified_image, cmap='gray')
        plt.title('Modified Image')

        plt.subplot(3, 2, 6)
        plt.plot(hist_gamma_corrected_modified, color='blue')
        plt.title('Modified Image Histogram')

    plt.tight_layout()
    plt.show()
