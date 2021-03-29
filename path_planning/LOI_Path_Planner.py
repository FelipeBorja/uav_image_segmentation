"""
Drone Path Planner
Hunter Ray
hura1154@colorado.edu

Last updated 3/25/21

This is the main file that plans the particular path for the low altitude inspector drone.
The path is planned using a wavefront planner where points are successively increased from the goal
to the drone's current location.

Inputs to this file are a numpy array of points with obstacles marked by a value of "1".
The output path is NOT considered optimal as it will simply go to the nearest point of interest (POI) from
the drone's current location. 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from Wavefront_functions import *

###  Inputs:
input_array = cv2.imread("../combined_detection/obstacle_image.png")
plotting = True  # Do you want to plot things?

# Take image array and break into boxes
res = 20
height, width, channels = input_array.shape  # input image dimensions
LOI_obj = LowDrone(height, width, res)  # same size as input image but only 1 channel
LOI_obj.drone = [5, 5]

### Start processing

for row in range(height):
    for col in range(width):
        check = input_array[row][col] == [1]
        if all(check):
            x = int(np.floor(row / res))
            y = int(np.floor(col / res))
            LOI_obj.map_array[x][y] = 1

# Add POI
LOI_obj.POI.append([80, 80])
LOI_obj.POI.append([60, 40])
LOI_obj.POI.append([95, 90])
LOI_obj.POI.append([90, 20])
base_points = copy.deepcopy(LOI_obj.POI)  # Copy points for plotting
path = []

# Calling Wavefront Planner for all points
for i in range(len(LOI_obj.POI)):
    # Find the next point to go to
    next_point, idx = LOI_obj.find_next_point()
    # Call planner
    path = wavefront_planner(LOI_obj.map_array, next_point, LOI_obj.drone, path)
    LOI_obj.POI.pop(idx)  # Remove POI from list
    LOI_obj.drone = next_point


### Plotting

# plt.imshow(LOI_obj.map_array)  # imshow gave weird results so all plotting is done with scatter plots
if plotting:
    height, width = LOI_obj.map_array.shape
    for row in range(height):
        for col in range(width):
            if LOI_obj.map_array[row][col]:
                plt.scatter(row, col, 10, c="red")
    for p in path:
        plt.scatter(p[0], p[1], 4, c="green")
    for p in base_points:
        plt.scatter(p[0], p[1], 20, c="blue")
    plt.scatter(5, 5, 10, c="orange")
    plt.show()

