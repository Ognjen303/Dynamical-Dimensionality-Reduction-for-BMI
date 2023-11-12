"""Compute eigenvalues and eigenvectors of A."""
import numpy as np
import cond_color
from modules import load_data
from q2_preprocessing import limit_psth, pca_dim_reduction
from q3_plot_pc_space_traj import plot_pc1_pc2_plane
from q4e_estimate_matrix_A import estimate_A


# Load data
X, times = load_data()

# Limit the PSTH to time interval between -150ms and +300ms
# Shape of X is (N x CT) = (12 x 4968) and T = 46
X, times, T = limit_psth(X, times)

# Dimensionality reduction by PCA
Z = pca_dim_reduction(X)  # shape is (M x CT)

M = Z.shape[0]
Z = Z.reshape(M, -1, T)

# Estimate the matrix A from actual data
A = estimate_A(Z)

K = A.shape[0]

# Find eigenvalues and eigenvectors of A
evals, evecs = np.linalg.eig(A)

# print(f'{evals=}')
print(f'{evals[:2]=}')
print(f'{evecs[:, :2]=}')

v = evecs[:, 0]
real_v, imag_v = v.real, v.imag

real_v_norm = real_v / np.linalg.norm(real_v)
imag_v_norm = imag_v / np.linalg.norm(imag_v)

P = np.zeros((2, K))
P[0, :] = real_v_norm
P[1, :] = imag_v_norm

print(f'{P.shape=}')

# Plot only from -150ms to +200ms
Z = Z[:, :, :-10]

P_FR = np.tensordot(P, Z, axes=1)
print(f'{P_FR.shape=}')

plot_pc1_pc2_plane(P_FR, savefig=False, T=36)
