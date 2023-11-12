"""Compute eigenvalues and eigenvectors of A."""
import numpy as np
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


def plt_2D_rotation_traj(A, Z, rot_plane=1, savefig=False):
    """
    For even M, every rotation in M-dimensional space can be decomposed into independent 2D
    rotations in M/2 orthogonal 2D planes (that only meet at the origin). In the case of our matrix A,
    which represents an infinitesimal rotation, these special 2D planes are related to the eigenvectors
    of A, as follows. The eigenvalues of an antisymmetric matrix are all pure imaginary, and come in
    complex conjugate pairs; in other words, they are all of the form ±jω for real and positive ω (where
    j = √−1). The pair of eigenvectors corresponding to a pair of complex conjugate eigenvalues are
    also complex conjugates of each other. It turns out that the real and imaginary parts of (either
    one) of these two eigenvectors constitute a pair of orthogonal real vectors that span one of the
    M/2 special 2D planes. The imaginary part of the corresponding eigenvalue (i.e., ω in the above
    notation) then provides the angular velocity of the rotation in that special 2D plane.

    Focus first on the eigenvalue with the largest imaginary part; this eigenvalue corresponds to the
    fastest special 2D rotation induced by A. Construct the 2 × M matrix P with its two rows given
    by the normalized real and imaginary parts of the eigenvector corresponding to this eigenvalue.
    (Note that the real and imaginary part vectors need to be first normalized by you to have unit
    length.) By applying P to Z obtain the special 2D projection of the M-dimensional trajectories,
    in the special plane with the fastest rotation. We will call this special plane the plane of fastest
    rotation, or FR plane for short, and will call the corresponding projection matrix P_FR.

    Parameters:
    A -> 2D Square antisymmetric matrix of shape MxM (M is even).
         Obtained from function estimate_A(Z).

    Z -> Data Matrix with 12 priciple components.

    rot_plane (int) -> indicates in which rotation plane
                       you want to visualise the trajectories. Takes value between 1 and M/2.
    """

    M = A.shape[0]

    if M % 2 != 0:
        raise ValueError(
            'M must be even, where M is shape of antisymmetric matrix A.')

    if rot_plane not in range(1, M//2):
        raise ValueError(
            '2D rotation plane you select must be in range 1 to {M//2}')

    # Find eigenvalues and eigenvectors of A
    evals, evecs = np.linalg.eig(A)

    eval = evals[2*(rot_plane-1)]  # Select eval corresponding to rot_plane
    omega = np.abs(eval.imag)
    print(f'{omega=:.4f}')

    v = evecs[:, 2*(rot_plane-1)]
    real_v, imag_v = v.real, v.imag

    real_v_norm = real_v / np.linalg.norm(real_v)
    imag_v_norm = imag_v / np.linalg.norm(imag_v)

    P = np.zeros((2, M))
    P[0, :] = real_v_norm
    P[1, :] = imag_v_norm

    # Plot only from -150ms to +200ms
    Z = Z[:, :, :-10]

    # Compute P_rot
    P_rot = np.tensordot(P, Z, axes=1)  # shape is 2 x C x T

    if rot_plane == 1:
        plt_title = "Trajectories in fastest rotation plane $P_{FR}$ with "

    elif rot_plane == 2:
        plt_title = "Trajectories in second fastest rotation plane with "

    elif rot_plane == 3:
        plt_title = "Trajectories in third fastest rotation plane with "

    plt_title += f"$\omega$={omega:.4f} for time range [-150ms, 200ms]"

    plot_pc1_pc2_plane(P_rot, plt_title, savefig=savefig, T=36, Q=5)


# To answer Qc
# To answer Q5d just set rot_plane=2 or 3
plt_2D_rotation_traj(A, Z, rot_plane=3, savefig=False)
