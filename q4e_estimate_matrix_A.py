"""Function to estimate matrix A from data Z."""
import numpy as np
import matplotlib.pyplot as plt
from modules import load_data, load_test_data
from q2_preprocessing import limit_psth, pca_dim_reduction
from q4_MLE_for_A import construct_H, reconstruct_A, construct_W, \
    construct_Q, construct_b, MLE_A, plot_A_matrix


def estimate_A(Z):
    """Function to estimate antisymmetric matrix A from data Z.

    Parameters:
    Z (ndarray): 3D data array of shape M x C x T.
                 Z is obtained by doing PCA dimensionality reduction
                 on data matrix X of PSTH recordings.

    Returns:
    A (ndarray): antisymmetric matrix of shape K x K,
                 where K = M(M-1)/2. It governs the autonomous dynamics of
                 the PSTH neural activity.
    """

    M, C, T = Z.shape

    # Take difference along time axis. The t+1-th column of dZ
    # denoted as dz_{t+1} in the report is dz_{t+1} = z_{t+1} - z_t:
    dZ = Z[:, :, 1:] - Z[:, :, :-1]  # shape is (M x C x (T-1))

    # Redefine Z by discarding the first column
    Z = Z[:, :, 1:]  # shape is (M x C x (T-1))

    # H is 3D array which linearly relates row vector beta and A
    H = construct_H(M)

    # W is 3D array of shape KxMxC(T-1). It linearly combines H and Z.
    W = construct_W(H, Z)

    Q = construct_Q(W)  # shape is KxK
    b = construct_b(dZ, W)  # shape is 1xK

    # Perfom Maximum likelihood estimate of A
    A = MLE_A(b, Q)

    return A


# Load data
X, times = load_data()
Z_test, A_test = load_test_data()

# Limit the PSTH to time interval between -150ms and +300ms
# Shape of X is (N x CT) = (12 x 4968) and T = 46
X, times, T = limit_psth(X, times)

# Dimensionality reduction by PCA
Z = pca_dim_reduction(X)  # shape is (M x CT)

# Reshape Z back into a 3D array of shape (M x C x T)
M = Z.shape[0]  # M = 12
Z = Z.reshape(M, -1, T)


# Estimate the antisymmetric matrix A of shape KxK
A_my_est = estimate_A(Z_test)

# plot_A_matrix(A_test, test=True)
# plot_A_matrix(A_my_est, test=False)

print(f'{A_test=}')  # stvarno resenje
print(f'{A_my_est=}') # moje resenje

print(f'Are the two matricies identical? Ans: \
      {np.allclose(A_my_est, A_test, atol=1e-4)}')
