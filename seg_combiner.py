"""
Image Segmentation Combiner
Felipe Borja
"""

import numpy as np
import cv2

""" Load images """
# Load and display images
building_input = cv2.imread('building_detection_seg.png')
road_input = cv2.imread('road_detection_seg.png')

height, width, channels = building_input.shape # input image dimensions
output_image = np.zeros((height, width, channels), np.uint8) # same size as input image
obstacle_array = np.zeros((height, width), np.uint8) # same size as input image but only 1 channel

for row in range(height):
    for col in range(width):
        # Get values of the road and building pixels at this row and column
        building_pixel = building_input[row][col]
        road_pixel = road_input[row][col]
        if(road_pixel.all() != 0):
            output_image[row][col] = [255,0,0]
        if(building_pixel.all() != 0):
            output_image[row][col] = [0,0,255]
            obstacle_array[row][col] = 255


cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output_image.png', output_image)
            
cv2.imshow('Obstacle Image', obstacle_array)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('obstacle_image.png', obstacle_array)   
