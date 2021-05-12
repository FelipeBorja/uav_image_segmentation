Path Planning Algorithm ReadMe

These functions implement a wavefront path planner to plan a sequential route through the area of operations to the
specified points of interest. The general layout for utilizing this script can be followed in "Path_Planner_call.py".
This script also contains an option to plot the final result. 

Input:
1. Obstacle Array: The planner takes in a numpy array that has obstacle points labeled as 1. 
2. Grid Resolution: This integer is input to allow the planner to discretize the grid. 
   It represents the size of each pixel. The final grid will be sized by dividing the height and width by the specified
   resolution.
3. Points of Interest: These POI should be provided in the form of a list of x,y coordinates: [[p1x,p1y],[p2x,p2y]]. 

Output:
The primary output is a set of GPS coordinates that the drone can fly to. 
The format will follow the input of the points of interest. 