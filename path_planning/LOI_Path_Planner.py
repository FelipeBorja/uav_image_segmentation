"""
Drone Path Planner
Hunter M. Ray
hura1154@colorado.edu

Last updated 3/31/21

This is the main file with function that plans the particular path for the low altitude inspector drone.
The path is planned using a wavefront planner where points are successively increased from the goal
to the drone's current location.


The output path is NOT considered optimal as it will simply go to the nearest point of interest (POI) from
the drone's current location. 
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy
from Wavefront_functions import *


###  Inputs:
def path_planner_init(input_array, poi_list,res,drone_start):
    """This function takes in an input array and list of points of interest (local frame).
    Returns the LOI_obj which stores the map at the appropriate resolution, points of interest,
    and drone location.
    Inputs to this file are a numpy array of points with obstacles marked by a value of "1"."""
    # input_array = cv2.imread("../combined_detection/obstacle_image.png")

    # Take image array and break into boxes
    height, width, channels = input_array.shape  # input image dimensions
    LOI_obj = LowDrone(height, width, res)  # same size as input image but only 1 channel
    LOI_obj.drone = drone_start

    ### Start processing
    # Discretize image
    for row in range(height):
        for col in range(width):
            check = input_array[row][col] == [1]
            if all(check):
                x = int(np.floor(row / res))
                y = int(np.floor(col / res))
                LOI_obj.map_array[x][y] = 1

    ## Add safety buffer:
    buffer = 3  # Increase this number to increase buffer around buildings
    for i in range(buffer):
        buffer_map = copy.deepcopy(LOI_obj.map_array)
        height,width = np.shape(buffer_map)
        for row in range(height):
            for col in range(width):
                check_up = 0
                check_down = 0
                check_left = 0
                check_right = 0
                if 0 <= row+1 < height:
                    check_up = buffer_map[row+1][col] == [1]
                if 0 <= row-1 < height:
                    check_down = buffer_map[row-1][col] == [1]
                if 0 <= col-1 < width:
                    check_left = buffer_map[row][col-1] == [1]
                if 0 <= col+1 < width:
                    check_right = buffer_map[row][col+1] == [1]
                if any([check_up,check_down,check_left,check_right]):
                    LOI_obj.map_array[row][col] = 1


    # Add POI
    if len(poi_list) == 0:
        LOI_obj.poi.append([80, 80])
        LOI_obj.poi.append([67, 34])
        LOI_obj.poi.append([99, 77])
        LOI_obj.poi.append([90, 20])
        LOI_obj.poi.append([90,58])
    else:
        LOI_obj.poi = poi_list

    LOI_obj.base_points = copy.deepcopy(LOI_obj.poi)  # Copy points for plotting
    return LOI_obj


def plan_next_point(loi_obj,path):
    """This function plans the path for the drone to the nearest point of interest.
    Function will add next path onto 'path' input"""
    # Calling Wavefront Planner
    # Find the next point to go to
    next_point, idx = loi_obj.find_next_point()
    # Call planner
    path = wavefront_planner(loi_obj.map_array, next_point, loi_obj.drone, path)
    loi_obj.poi.pop(idx)  # Remove POI from list
    loi_obj.drone = next_point

    return path
