import numpy as np
import matplotlib.pyplot as plt
from modules import load_data, save_fig
from q2_preprocessing import limit_psth, pca_dim_reduction


# Load data
X, times = load_data()

# Limit the PSTH to time interval between -150ms and +300ms
# Shape of X is (N x CT) = (12 x 4968) and T = 46
X, times, T = limit_psth(X, times)

# Dimensionality reduction by PCA
Z = pca_dim_reduction(X)  # shape is (M x CT)

M = Z.shape[0]  # M = 12
C = int(Z.shape[1] / T)  # C = 108


# ---------------- Q4(a) Log-likelihood and its (naive) gradient ----------------


# Reshape Z back into a tensor of shape (M x C x T)
Z = Z.reshape(M, -1, T)

# Take difference along time axis. The t+1-th column of dZ
# denoted as dz_{t+1} in the report is dz_{t+1} = z_{t+1} - z_t:
dZ = Z[:, :, 1:] - Z[:, :, :-1]  # shape is (M x C x (T-1))
# print(f'{dZ.shape=}')

# Redefine Z by discarding the first column
Z = Z[:, :, 1:]  # shape is (M x C x (T-1))
# print(f'{Z.shape=}')

# Log-likelihood, log p(Z | sigma=1, A)
# ll = - 0.5 * Z.T @ A.T @ A @ Z + dZ.T @ A @ Z + const

# Naive gradient of ll w.r.t. A:
# dll = -A @ Z @ Z.T + dZ @ Z.T

# Maximum likelihood estimate of an unconstrained A:
# set dll = 0 --> A = dZ @ Z.T @ (Z @ Z.T)^{-1}

# ---------------- Q4(b) Parametrising an antisymmetric A ----------------


def index_to_pair(a, M):
    """
    Given an index a in the range 0 to K-1 (where K = M(M-1)/2),
    corresponding to the a-th entry of vector beta, returns the pair
    of corresponding indices (i, j) in matrix A such that Aij = beta[a],
    where i < j.

    Parameters:
    - a (int): Index in the range 0 to K-1.
    - M (int): Size of the square matrix A.

    Returns:
    - tuple: A tuple (i, j) representing the indices in matrix A.

    Raises:
    - ValueError: If the provided index is out of range.
    """
    if not (0 <= a < M * (M - 1) / 2):
        raise ValueError("Index out of range.")

    # Find i, j such that Aij = beta[a]
    i = 0
    j = 1
    count = 0

    while count < a:
        j += 1
        if j == M:
            i += 1
            j = i + 1
        count += 1

    return i, j


def construct_H(M):
    """
    Constructs the H tensor. Since the elements of antisymmetric matrix A
    are linearly related to row vector beta of length K, we can write
    the elements of A as a linear combination of beta[a] acording to:

    A[i, j] = sum_{a=1}^{K} beta[a] * H[a, i, j]

    Elements H[a, i, j] take values in {-1, 0, +1}.

    Parameters:
    - M (int): Size of the square matrix A.

    Returns:
    - H (numpy.ndarray) of shape KxMxM: Linearly relates A and beta.
    """

    # K is number of unconstrained entries in A
    K = M * (M - 1) // 2

    H = np.zeros((K, M, M), dtype=int)

    for a in range(K):
        i, j = index_to_pair(a, M)

        H[a, i, j] = 1
        H[a, j, i] = -1

    return H


def reconstruct_A(beta, H):
    """
    Reconstruct antisymmetric matrix A. Basically does this:

    for a in range(K):
        A += beta[:, a] * H[a, :, :]

    Parameters:
    - beta: row vector of shape (1xK)
    - H: tensor of shape (KxMxM)

    Returns:
    - A: antisymmetric matrix with shape (MxM) filled with
         entries of beta in a row major order.
    """

    # Ovo se uopsteno naziva:
    # KONTRAKCIJA PO PONOVLJENOM INDEKSU

    # np.tensordot(beta, H, axes=1) je specijalan slucaj ovoga,
    # koji se jos zove i "tenzorski skalarni proizvod".

    _, M = beta.shape

    if M != H.shape[0]:
        raise ValueError("Dimensions don't align up.")

    K, M, M = H.shape

    if K != M*(M-1)/2:
        raise ValueError("Check the dimensions of H matrix, they are wrong.")

    A = np.tensordot(beta, H, axes=1)
    return np.squeeze(A, axis=0)


# print(f'{M=}')
H = construct_H(M)


# ---------------- Q4(c) Gradient w.r.t. beta ----------------


# Now that we can represent out A matrix using beta
# We can rewrite the Log-likelihood, log p(Z | sigma=1, A)
# in terms of beta log p(Z | sigma=1, beta)

def construct_W(H, Z):
    """
    Parameters:
    H -> array of shape KxMxM
    Z -> array of shape MxCx(T-1)

    Returns:
    W -> Array of shape KxMxC(T-1)"""

    K, M, M = H.shape

    if K != M*(M-1) / 2 or M != Z.shape[0]:
        raise ValueError(
            "Check the dimensions of H and Z matrix, they are wrong.")

    Z = Z.reshape(M, -1)  # shape is M x C(T-1)

    return np.tensordot(H, Z, axes=1)


W = construct_W(H, Z)
# print(f'{W.shape=}')

# Log-likelihood log p(Z | sigma=1, beta) is:
# ll = -0.5 * [np.tensordot(beta, W, axes=1).T @ np.tensordot(beta, W, axes=1)
#              - 2 * dZ.T @ np.tensordot(beta, W, axes=1)]

# gradient of ll w.r.t. beta is:
# dll = -beta @ np.tensordot()


def construct_Q(W):
    """
    Parameter:
    W -> 3D array of shape K x M x C(T-1)

    Returns:
    Q -> 2D array of shape K x K. We get Q by contracting W with
         itself along 1 and 2 axis.
    """
    return np.tensordot(W, W, axes=([1, 2], [1, 2]))


Q = construct_Q(W)
# print(f'{Q.shape=}')


def construct_b(dZ, W):
    """
    Parameter:
    dZ -> 3D array of shape M x C x (T-1)
    W ->  3D array of shape K x M x C(T-1)

    Returns:
    b -> Row vector of shape 1 x K. It is calculated by contracting
         reshaped version of dZ and W.
    """

    # Reshape dZ
    dZ = dZ.reshape(M, -1)  # shape is MxC(T-1)
    # print(f'{dZ.shape=}')

    # Compute the row vector b
    b = np.tensordot(dZ, W, axes=([0, 1], [1, 2]))
    b = b.reshape((1, 66))
    # print(f'{b.shape=}')
    return b


b = construct_b(dZ, W)

# The gradient of Log-likelihood w.r.t beta is:
# b - beta @ Q


# ---------(d) Maximum Likelihood Estimate for antisymetric A ---------


def MLE_A(b, Q):
    """Performs Maximum Likelihood Estimate of antisymmetric matrix A."""

    # For MLE set grad of ll w.r.t.b equal to 0, which gives:
    # b - beta @ Q = 0. Hence

    # print(f'{Q.T.shape=}')
    # print(f'{b.T.shape=}')
    beta_transpose = np.linalg.solve(Q.T, b.T)

    beta = beta_transpose.T

    # print(f'{beta.shape=}')

    A = reconstruct_A(beta, H)

    # print(f'{A.shape=}')
    # print(f'{A=}')

    return A


A = MLE_A(b, Q)


def plot_A_matrix(A, test=False):

    # Plot entries of anitysymmetric matrix A

    fig, ax = plt.subplots(figsize=(10, 8))
    pos = ax.imshow(A, cmap='Blues', interpolation='none')
    fig.colorbar(pos, ax=ax)

    # if test:  # If we passed the actual solution matrix A
    #     ax.set_title('Antisymmetric matrix A_test')
    #     save_fig(fig, 'Q4e_A_test_colour_plot')
    # else:
    #     ax.set_title(f'MLE of antisymmetric matrix A')
    #     save_fig(fig, 'Q4e_A_colour_plot')

    plt.show()
