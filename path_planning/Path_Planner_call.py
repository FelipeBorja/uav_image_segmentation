from LOI_Path_Planner import *

"""Use this file to directly call the path planner. Implementation into larger framework can follow this general idea.
"""
input_array = cv2.imread("../combined_detection/obstacle_image.png")

#Create initial object
# How big is each pixel? Resulting grid must have ~0.5m squares to be within drone's position uncertainty
resolution = 20
obj = path_planner_init(input_array, [],resolution)

#Plan path
path = []
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

