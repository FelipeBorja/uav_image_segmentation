import cv2
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

flags = [i for i in dir(cv2) if i.startswith('COLOR_')]

input_image = cv2.imread('input_image.png')
plt.imshow(input_image)
plt.show()

""" Accurate coloring """
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.imshow(input_image)
plt.show()

""" Make HSV image """
input_hsv = cv2.cvtColor(input_image, cv2.COLOR_RGB2HSV)

h, s, v = cv2.split(input_hsv)
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1, projection="3d")

pixel_colors = input_image.reshape((np.shape(input_image)[0]*np.shape(input_image)[1], 3))
norm = colors.Normalize(vmin=-1.,vmax=1.)
norm.autoscale(pixel_colors)
pixel_colors = norm(pixel_colors).tolist()

axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
axis.set_xlabel("Hue")
axis.set_ylabel("Saturation")
axis.set_zlabel("Value")
plt.show()

""" Decide color range """
light_grey = (120, 120, 120)
dark_grey = (180, 180, 180)

""" Make mask """
mask = cv2.inRange(input_hsv, light_grey, dark_grey)
# To impose the mask on top of the original image, you can use cv2.bitwise_and(), 
# which keeps every pixel in the given image if the corresponding value in the mask is 1:
result = cv2.bitwise_and(input_image, input_image, mask=mask)

""" See Results """
plt.subplot(1, 2, 1)
plt.imshow(mask, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.show()