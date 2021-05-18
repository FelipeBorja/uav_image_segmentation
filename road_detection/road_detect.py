"""
Quick and Drity Color-Based Road Detection
Felipe Borja
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

""" Lower and upper RGB limits """
lower  = 120
upper = 240

""" Load images """
# Load and display images
input_image = cv2.imread('neighborhood_image.jpg')
plt.imshow(input_image)
plt.show()

height, width, channels = input_image.shape # input image dimensions
output_image  = np.zeros((height, width, channels), np.uint8) # same size as input image

for row in range(height):
    for col in range(width):
        # print("Row: " + str(row) + ", Column: " + str(col))
        # Get BGR values of the pixel at this row and column
        pixel = input_image[row][col]
        pix_B = pixel[0]
        pix_G = pixel[1]
        pix_R = pixel[2]
        # Check to see if pixel is gravel grey according to BGR conditions
        if(pix_B < upper and pix_B > lower and pix_G < upper and pix_G > lower and pix_R < upper and pix_R > lower): # TODO: make this more dynamic and less static
            output_image[row][col] = pixel


""" Run some image gradient analyses using built-in OpenCV functions"""
laplacian = cv2.Laplacian(input_image,cv2.CV_64F)
sobelx = cv2.Sobel(input_image,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(input_image,cv2.CV_64F,0,1,ksize=5)
# TODO: incorporate the filters into the final road estimation (one or more filters)

plt.imshow(output_image)
plt.show()
plt.imshow(laplacian)
plt.show()
plt.imshow(sobelx)
plt.show()
plt.imshow(sobely)
plt.show()

cv2.imwrite('output_image_neighborhood.png', output_image)
