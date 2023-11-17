"""Control Analysis"""
import numpy as np
import matplotlib.pyplot as plt
from modules import load_data
from q2_preprocessing import limit_psth, pca_proj_matrix, pca_dim_reduction
from q4e_estimate_matrix_A import estimate_A
from q5 import *


# Load data
X, times = load_data()

# Number of neurons, contitions and timebins
N, C, _ = X.shape

# Pick the bin corresponding to time -150ms
t0 = 65
print(f'{times[t0]=}')

# Distort the raw PHTS
for n in range(N):

    # Randomly pick conditions which to flip for each neuron
    c = np.random.choice(C, (C//2,), replace=False)

    # Inversion of subsequent PSTH values about X[n, c, t0]
    X[n, c, t0:] = 2 * np.expand_dims(X[n, c, t0], axis=1) - X[n, c, t0:]

# Create figure and axis
plt.plot(times, X[0, 0, :])

plt.show()


# --------- Rerun the computational steps from q2 to q5c
# --------- and plot fastest rotation plane

# Limit the PSTH to time interval between -150ms and +300ms
# Shape of X is (N x CT) = (182 x 4968) and T = 46
X, times, T = limit_psth(X, times)

# Obtain the PCA Projection matrix V_M
V_M = pca_proj_matrix(X)

# Dimensionality reduction by PCA
Z = pca_dim_reduction(X)  # shape is (M x CT)

M = Z.shape[0]
Z = Z.reshape(M, -1, T)

# Estimate the matrix A from actual data
A = estimate_A(Z)

# Construct the projection matrix P for fastest rotation plane
P_FR, omega = construct_P(A, rot_plane=1)

# Trajectories are obtained by projecting
# data Z onto the fastest rotation plane
P_rot = project_movement_to_rotation_plane(P_FR, Z)

# Plot the trajectories in the fastest rotation plane
plt_2D_rotation_traj(P_rot, omega, rot_plane=1, savefig=True)
