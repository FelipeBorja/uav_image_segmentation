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
obj = path_planner_init(input_array, points_of_interest,resolution)

#Plan path
path = []  # Start path
for i in range(len(obj.poi)):
    path = plan_next_point(obj,path)  # Function recursively adds onto path as necessary

plotting = 1  # Do you want to show the path?
if plotting:
    height, width = obj.map_array.shape
    plt.imshow(obj.map_array)
    for p in path:
        plt.scatter(p[1], p[0], 4, c="green")
    for p in obj.base_points:
        plt.scatter(p[1], p[0], 20, c="blue")
    plt.scatter(5, 5, 10, c="orange")
    plt.show()

