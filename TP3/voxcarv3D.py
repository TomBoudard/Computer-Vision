import matplotlib.image as mpimg
import numpy as np
from skimage import measure
import trimesh


# Camera Calibration for Al's image[1..12].pgm   
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [ 78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
     0.525731, 0, -0.85065, 2],
    [0, 33.6163, -230.924, 300, -178.763, 127.597, -78.8596, 300,
     0, 0.85065, -0.525731, 2],
    [-78.8596, -178.763, 127.597, 300, 73.2053, 0, 221.578, 300,
     -0.525731, 0, 0.85065, 2],
    [78.8596, -178.763, 127.597, 300, 230.924, 0, 33.6163, 300,
     0.525731, 0, 0.85065, 2],
    [0, -221.578, -73.2053, 300, 178.763, -127.597, 78.8596, 300,
     0, -0.85065, 0.525731, 2],
    [0, 33.6163, 230.924, 300, 178.763, 127.597, 78.8596, 300,
     0, 0.85065, 0.525731, 2],
    [-33.6163, -230.924, 0, 300, -127.597, -78.8596, 178.763, 300,
     -0.85065, -0.525731, 0, 2],
    [-221.578, -73.2053, 0, 300, -127.597, 78.8596, 178.763, 300,
     -0.85065, 0.525731, 0, 2],
    [221.578, -73.2053, 0, 300, 127.597, 78.8596, -178.763, 300,
     0.85065, 0.525731, 0, 2],
    [33.6163, -230.924, 0, 300, 127.597, -78.8596, -178.763, 300,
     0.85065, -0.525731, 0, 2]
])


# Build 3D grids
# 3D Grids are of size: resolution x resolution x resolution/2
resolution = 100
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]

# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)
 

# ---------- MAIN ----------
if __name__ == "__main__":
    
    # read the input silhouettes
    for i in range(12):
        myFile = "image{0}.pgm".format(i)
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        # Compute grid projection in images
        # Update grid occupancy
        M = np.reshape(calib[i], (3, 4))
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution//2):
                    if occupancy[i][j][k] == 0:
                        continue
                    x, y, z = X[i, j, k], Y[i, j, k], Z[i, j, k]
                    proj = np.matmul(M, np.array([[x], [y], [z], [1]]))
                    u = int(proj[0][0]//proj[2][0])
                    v = int(proj[1][0]//proj[2][0])
                    if (0 <= u < img.shape[0] and 0 <= v < img.shape[1]):
                        if (img[u][v] == 0):
                            occupancy[i][j][k] = 0

    # Voxel visualization

    # Use the marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(occupancy, 0.25)

    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alvoxels.off')
 
#Positional encoding