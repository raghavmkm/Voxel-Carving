import scipy.io
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
plt.rc('figure', max_open_warning = 0)


# Load camera matrices
# Here we are just reading in some important information for the project. The intrinsic and extrinsic properties of the camera settings are read in from the dino_Ps.mat file which contains the 
# camera/projection matrix values.
# We also read in the images of the object we are trying to reconstruct. Here it is that of the dinosaur. 
# Output :
# projections: the projection matrices of the each single camera angle/snapshot based on the properties of the camera and that of the external conditions of the photo
# images: the 36 snapshots of the dinosaur image
# height: the height of the images
# width: the width of the images
def read_input():
    projection_matrix = scipy.io.loadmat("input/dino_Ps.mat")
    projection_matrix = projection_matrix["P"]
    projections = [projection_matrix[0, i] for i in range(projection_matrix.shape[1])]

    # load images
    files = sorted(glob.glob("input/*.ppm"))
    images = []
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED).astype(float)
        image /= 255
        images.append(image[:, :, ::-1])
    height, width, __ = images[0].shape
    # for im in images:
    #     plt.imshow(im)   
    #     plt.show()
    return projections, images, height, width

# Here we get silouhettes of the images.
# We do this so that we know the distinction between the object and the background. This is essential for carving out the object later.
# The image in each camera is converted to a binary image using the blue-screen background and some morphological operators to clean up the images like removing holes. 
# Holes in this mask are particularly dangerous as they will cause voxels to be carved away that shouldn't be - we can end up drilling a hole through the object!
# Input:
# images: the images/snapshots taken of the dinosaur
# Output:
# silhouettes: the masked and processes binary images
def silhouette_images(images):
    silhouettes = []
    for image in images:
        #splitting the image into a binary image separating the object from the background
        temp = np.abs(image - [0.0, 0.0, 0.75])
        temp = np.sum(temp, axis=2)
        y, x = np.where(temp <= 1.1)
        image[y, x, :] = [0.0, 0.0, 0.0]
        image = image[:, :, 0]
        image[image > 0] = 1.0
        image = image.astype(np.uint8)
        #removing the holes in the image
        kernel = np.ones((5, 5), np.uint8)
        im = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel) #removes holes which is a clean up that is required
        silhouettes.append(image)
        # plt.imshow(im)
        # plt.show()
    return silhouettes

# This function creates a regular 3D grid of voxel elements ready for carving away based on the imput images properties. This is one of the first steps taken in most Voxel carving algorithms.
# Output:
# grid: The voxel grid
def init_grid():
    s = 120
    x, y, z = np.mgrid[:s, :s, :s]
    grid = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
    grid = grid.T
    grid_shape = grid.shape[0]
    xmax, ymax, zmax = np.max(grid, axis=0)
    grid[:, 0] /= xmax
    grid[:, 1] /= ymax
    grid[:, 2] /= zmax
    mean = grid.mean(axis=0)
    grid -= mean
    grid /= 5
    grid[:, 2] -= 0.62
    grid = np.vstack((grid.T, np.ones((1, grid_shape))))
    return grid

# This is the final function where we implement the carving based on the conditions of the voxel.
# The algorithm carves out voxels that are not in the silhouettes contained in the camera.
# Inputs:
# projections: the projection matrices of the each single camera angle/snapshot based on the properties of the camera and that of the external conditions of the photo
# silhouettes: the masked and processes binary images
# grid: The voxel grid
# height: the height of the images
# width: the width of the images
# Output:
# occuoancy: number of cameras where voxel seems to be mapped to
# voxels: the carved grid
def carve(projections, silhouettes, grid, height, width):
    filled = []
    for projection, image in zip(projections, silhouettes):
        # print(projection)
        # print(grid)
        uvs = projection @ grid
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        #condition for whether the voxel needs to be carved out
        x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < width)
        y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < height)
        good = np.logical_and(x_good, y_good)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = image[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res 
        filled.append(fill)
    filled = np.vstack(filled)
    # the occupancy is computed as the number of camera in which the point "seems" not empty
    occupancy = np.sum(filled, axis=0)
    # Select occupied voxels
    voxels = grid.T
    return occupancy, voxels

