import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os 

# generate random 3D point cloud
num_points = 100
cloud = np.random.rand(num_points, 3)

# create figure and axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

import pickle

# your save_faces_path

your_save_faces_path = '..'
your_save_faces_path = os.path.join(your_save_faces_path,'dataX.pkl')

# Open the pickle file in read mode
with open(your_save_faces_path, 'rb') as f:
    # Load the vector from the pickle file
    my_vector = pickle.load(f)

# Now you can use the vector as a normal Python object
print(my_vector.shape)

vec = my_vector[3]

# plot point cloud as a scatter plot
ax.scatter(vec[0,:], vec[1,:], vec[2,:], c='b', marker='o')

# set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# allow for interactive rotation and zooming
ax.view_init(elev=30, azim=120)
ax.dist = 8  # set initial distance to the point cloud

# create a callback function to handle zoom events
def on_scroll(event):
    # get current distance to the point cloud
    current_dist = ax.dist
    # compute the new distance based on the scroll event
    if event.button == 'up':
        new_dist = current_dist * 0.9
    elif event.button == 'down':
        new_dist = current_dist * 1.1
    else:
        new_dist = current_dist
    # set the new distance and redraw the plot
    ax.dist = new_dist
    fig.canvas.draw_idle()

# connect the callback function to the mouse scroll event
fig.canvas.mpl_connect('scroll_event', on_scroll)

plt.show()

