import math as m
import matplotlib.image as mpimg
import numpy as np
import skimage
from skimage import measure
import torch
from torch import nn
import trimesh
import random as rd

# Camera Calibration for Al's image[1..12].pgm
calib = np.array([
    [-78.8596, -178.763, -127.597, 300, -230.924, 0, -33.6163, 300,
     -0.525731, 0, -0.85065, 2],
    [0, -221.578, 73.2053, 300, -178.763, -127.597, -78.8596, 300,
     0, -0.85065, -0.525731, 2],
    [78.8596, -178.763, -127.597, 300, -73.2053, 0, -221.578, 300,
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

# Training
MAX_EPOCH = 10
BATCH_SIZE = 100

# Build 3D grids
# 3D Grids are of size resolution x resolution x resolution/2
resolution = 100
step = 2 / resolution

# Voxel coordinates
X, Y, Z = np.mgrid[-1:1:step, -1:1:step, -0.5:0.5:step]
# Random Voxel coordinates
nb_triplet_train = 1000000



# Voxel occupancy
occupancy = np.ndarray((resolution, resolution, resolution // 2), dtype=int)

# Voxels are initially occupied then carved with silhouette information
occupancy.fill(1)


# MLP class
class MLP(nn.Module):
    """
    Multilayer Perceptron.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 120),
            nn.Tanh(),
            nn.Linear(120, 240),
            nn.ReLU(),
            nn.Linear(240, 240),
            nn.ReLU(),
            nn.Linear(240, 120),
            nn.ReLU(),
            nn.Linear(120, 60),
            nn.ReLU(),
            nn.Linear(60, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """ Forward pass """
        return self.layers(x)


# GPU or not GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

    
# MLP Training
def nif_train(data_in, data_out, batch_size):
    # Initialize the MLP
    mlp = MLP()
    mlp = mlp.float()
    mlp.to(device)

    # Normalize cost between 0 and 1 in the grid
    n_one = (data_out == 1).sum()

    # loss for positives will be multiplied by this factor in the loss function
    p_weight = (data_out.size()[0] - n_one) / n_one
    print("Pos. Weight: ", p_weight)

    # Define the loss function and optimizer
    # loss_function = nn.CrossEntropyLoss()

    # sigmoid included in this loss function
    loss_function = nn.BCEWithLogitsLoss(pos_weight=p_weight)
    optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-2)

    # Run the training loop
    for epoch in range(0, MAX_EPOCH):

        print(f'Starting epoch {epoch + 1}/{MAX_EPOCH}')

        # Creating batch indices
        permutation = torch.randperm(data_in.size()[0])

        # Set current loss value
        current_loss = 0.0
        accuracy = 0

        # Iterate over batches
        for i in range(0, data_in.size()[0], batch_size):

            indices = permutation[i:i + batch_size]
            batch_x, batch_y = data_in[indices], data_out[indices]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Zero the gradient
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(batch_x.float())

            # Compute loss
            loss = loss_function(outputs, batch_y.float())

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print current loss so far
            current_loss += loss.item()
            if (i/batch_size) % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      ((i/batch_size) + 1, current_loss / (i/batch_size) + 1))

        outputs = torch.sigmoid(mlp(data_in.float()))
        acc = binary_acc(outputs, data_out)
        print("Binary accuracy: ", acc)

        # Training is complete.
    print('MLP trained.')
    return mlp


# IOU evaluation between binary grids
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(y_pred)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    accuracy = correct_results_sum / y_test.shape[0]
    accuracy = torch.round(accuracy * 100)
    return accuracy


def main():
    # Generate X,Y,Z and occupancy
    # Regular grid
    for imgId in range(12):
        myFile = "image{0}.pgm".format(imgId)
        print(myFile)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        M = np.reshape(calib[imgId], (3, 4))
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

    # Format data for PyTorch
    data_in = np.stack((X, Y, Z), axis=-1)
    resolution_cube = resolution * resolution * resolution
    data_in = np.reshape(data_in, (resolution_cube // 2, 3))
    data_out = np.reshape(occupancy, (resolution_cube // 2, 1))

    nbIn, nbOut = 0, 0
    nbIn = (data_out == 1).sum()
    nbOut = data_out.shape[0] - nbIn

    if nbIn > nbOut:
        nbPointToDel = nbIn - nbOut
        for i in range(nbPointToDel):
            index = np.argwhere(data_out == np.array([1]))
            randomIdIndex = rd.randint(0, index.shape[0]-1)
            randomIndex = index[randomIdIndex][0]

            data_out = np.delete(data_out, randomIndex, axis=0)
            data_in = np.delete(data_in, randomIndex, axis=0)
    elif nbIn < nbOut:
        nbPointToDel = nbOut - nbIn
        for i in range(nbPointToDel):
            index = np.argwhere(data_out == np.array([0]))
            randomIdIndex = rd.randint(0, index.shape[0]-1)
            randomIndex = index[randomIdIndex][0]

            data_out = np.delete(data_out, randomIndex, axis=0)
            data_in = np.delete(data_in, randomIndex, axis=0)



            
    # Random grid
    # data_in = np.array([[rd.random()*2-1, rd.random()*2-1, rd.random()-0.5] for _ in range(nb_triplet_train)])

    # imageMatrixList = []
    # for i in range(12):
    #     myFile = "image{0}.pgm".format(i)
    #     img = mpimg.imread(myFile)
    #     if img.dtype == np.float32:  # if not integer
    #         img = (img * 255).astype(np.uint8)

    #     M = np.reshape(calib[i], (3, 4))
    #     imageMatrixList.append((img, M))
    
    # # Format data for PyTorch
    # data_in = np.stack((X, Y, Z), axis=-1)
    # resolution_cube = resolution * resolution * resolution
    # data_in = np.reshape(data_in, (resolution_cube // 2, 3))
    # data_out = np.reshape(occupancy, (resolution_cube // 2, 1))

    imageMatrixList = []
    for i in range(12):
        myFile = "image{0}.pgm".format(i)
        img = mpimg.imread(myFile)
        if img.dtype == np.float32:  # if not integer
            img = (img * 255).astype(np.uint8)

        M = np.reshape(calib[i], (3, 4))
        imageMatrixList.append((img, M))

    # data_in = np.array([[rd.random()*2-1, rd.random()*2-1, rd.random()-0.5] for _ in range(nb_triplet_train)])
    data_in = []
    data_out = []
    cpt_in = 0
    cpt_out = 0
    while len(data_in) < nb_triplet_train:
    # for x, y, z in data_in:
        x, y, z = rd.random()*2-1, rd.random()*2-1, rd.random()-0.5
        for img, M in imageMatrixList:
            proj = np.matmul(M, np.array([[x], [y], [z], [1]]))
            u = int(proj[0][0]//proj[2][0])
            v = int(proj[1][0]//proj[2][0])
            if (0 <= u < img.shape[0] and 0 <= v < img.shape[1]):
                if (img[u][v] == 0):
                    # data_out.append([0])
                    if (cpt_in < nb_triplet_train // 2):
                        cpt_in += 1
                        data_in.append([x, y, z])
                        data_out.append([0])
                    break
        else:
            # data_out.append([1])
            if (cpt_out < (nb_triplet_train+1) // 2):
                cpt_out += 1
                data_in.append([x, y, z])
                data_out.append([1])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    print(data_in.shape)
    print(data_out.shape)

    # Pytorch format
    data_in = torch.from_numpy(data_in).to(device)
    data_out = torch.from_numpy(data_out).to(device)

    # Train mlp
    mlp = nif_train(data_in, data_out, BATCH_SIZE)  # data_out.size()[0])

    data_in = np.stack((X, Y, Z), axis=-1)
    resolution_cube = resolution * resolution * resolution
    data_in = np.reshape(data_in, (resolution_cube // 2, 3))

    data_in = torch.from_numpy(data_in).to(device)

    # Visualization on training data
    outputs = mlp(data_in.float())
    occ = outputs.detach().cpu().numpy()  # from torch format to numpy

    # Go back to 3D grid
    newocc = np.reshape(occ, (resolution, resolution, resolution // 2))
    newocc = np.around(newocc)

    # print(newocc)

    # Marching cubes
    verts, faces, normals, values = measure.marching_cubes(newocc, 0.25)
    # Export in a standard file format
    surf_mesh = trimesh.Trimesh(verts, faces, validate=True)
    surf_mesh.export('alimplicit.off')


# --------- MAIN ---------
if __name__ == "__main__":
    main()
