from LOI_Path_Planner import *

"""Use this file to directly call the path planner. Implementation into larger framework can follow this general idea.
"""
#The input array is the image that is segmented to detect obstacles
input_array = cv2.imread("../combined_detection/obstacle_image.png")

#Create initial object
#The output of this function is a grid that is discretized into areas where there are and are not buildings. The points
# of interest have also been marked within this grid.
resolution = 20  # How big is each pixel? Resulting grid must have ~0.5m squares
points_of_interest = []  # What points have been selected for visitation
drone_starting_location = [5,5]
obj = path_planner_init(input_array, points_of_interest,resolution,drone_starting_location)

#Plan path
path = []  # Start path
for i in range(len(obj.poi)):
    path = plan_next_point(obj,path)  # Function recursively adds onto path as necessary

plotting = 1  # Do you want to show the path?
# resolution = 1
if plotting:
    height, width = obj.map_array.shape
    raw_image = cv2.imread("../building_detection/odm_orthophoto_cropped.jpg")
    plt.imshow(raw_image)
    # plt.imshow(obj.map_array)

    for p in path:
        plt.scatter(p[1]*resolution, p[0]*resolution, 4, c="red")
    for p in obj.base_points:
        plt.scatter(p[1]*resolution, p[0]*resolution, 20, c="blue")
    plt.scatter(5*resolution, 5*resolution, 30, c="red")
    plt.show()

