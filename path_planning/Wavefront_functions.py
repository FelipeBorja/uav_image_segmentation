"""Classes used in defining the map"""
import numpy as np
import matplotlib.pyplot as plt
import copy


class LowDrone:

    def __init__(self, height, width, resolution):
        self.map_array = np.zeros((int(np.ceil(height / resolution)), int(np.ceil(width / resolution))), np.uint8)
        self.poi = []
        self.drone = []
        self.resolution = resolution
        self.base_points = []  # Copies poi from initial configuration

    def find_next_point(self):
        """This function finds the next nearest points. Only linear distance is estimated"""
        dist = 10000
        best_p = []
        idx = 0
        for i in range(len(self.poi)):
            point = self.poi[i]
            # p_dist = np.linalg.norm(point, self.drone)
            p_dist = np.sqrt((point[0] - self.drone[0]) ** 2 + (point[1] - self.drone[1]) ** 2)
            if p_dist < dist:
                best_p = point
                dist = p_dist
                idx = i
        return best_p, idx


def addNodes(grid, count, current, neighbors):
    """This function adds nodes that are adjacent to the current node.
    Input:
    grid is a numpy array with values at each point
    current is a list of nodes that are continually added to
    count is the current number of steps away from the goal
    neighbors is a list of all possible actions that can be taken

    Output:
    Function returns the modified grid and children, which are the new leaf nodes that have been created"""
    children = []
    for c in range(len(current)):  # For each node that has been fed in
        for n in range(len(neighbors)):  # For each possible direction that can be taken
            leaf = [current[c][0] + neighbors[n][0], current[c][1] + neighbors[n][1]]
            if 0 <= leaf[0] < grid.shape[0]:  # Check if in grid space
                if 0 <= leaf[1] < grid.shape[1]:  # Check if in grid space
                    val = grid[leaf[0]][leaf[1]]
                    if val == 0:  # Grid value must be zero. Value of 1 is an obstacle
                        if val != count + 1:  # If grid location hasn't been updated yet
                            children.append(leaf)
                            grid[leaf[0]][leaf[1]] = count + 1

    return grid, children


def checkNN(grid, current, motions):
    """This function takes in the modified grid and finds the incrementally lowest grid value.
    This value is the next step in the path and is then returned to the higher function."""
    new_point = []
    now_value = grid[current[0]][current[1]]
    for n in range(len(motions)):
        leaf = [current[0] + motions[n][0], current[1] + motions[n][1]]
        if 0 <= leaf[0] < grid.shape[0]:  # Check if in grid space
            if 0 <= leaf[1] < grid.shape[1]:  # Check if in grid space
                if grid[leaf[0]][leaf[1]] == now_value - 1:
                    new_point = leaf

    return new_point


def wavefront_planner(grid_def, goal, start, path):
    """This function runs a wavefront planner to find a path along a grid from a goal to start location.
    The system is currently setup to move to any adjacent and diagonal node"""
    up = [0, 1]
    down = [0, -1]
    left = [-1, 0]
    right = [1, 0]
    downr = [1, -1]
    downl = [-1, -1]
    upr = [1, 1]
    upl = [-1, 1]
    # motions = [up, down, left, right, downr, downl, upr, upl]
    motions = [downr, downl, upr, upl,up, down, left, right]
    grid = copy.deepcopy(grid_def)
    current = [goal]
    count = 2
    grid[goal[0]][goal[1]] = count

    # Wave propagation from goal to current drone location
    while grid[start[0]][start[1]] == 0:  # Expand points until starting location has been covered
        grid, current = addNodes(grid, count, current, motions)
        count += 1

    if len(path) == 0:  # If no path has been started yet
        path = [start]

    # Debugging Plots
    # plt.imshow(grid)
    # plt.show()

    # Plan Path by moving downhill the
    current = start
    while current != goal:
        move = checkNN(grid, current, motions)
        path.append(move)
        current = move

    return path
