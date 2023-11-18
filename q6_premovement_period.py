"""Applies projections obtained from interval [-150ms, 300ms]
to the interval [-800ms, -150ms],
which is refered to as pre-movement period.
"""
import matplotlib.pyplot as plt
import cond_color
from modules import load_data, save_fig
from q2_preprocessing import limit_psth, pca_proj_matrix, pca_dim_reduction
from q4e_estimate_matrix_A import estimate_A
from q5 import *

# ------------------ Q5 REPEATED ----------------
# Lower you will find the start of Q6, but this Q5 REPEATED
# code is needed we can answer Q6

# Load data
X, times = load_data()

# Limit the PSTH to time interval [-150ms, +300ms]
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


# ------------------ START OF Q6 ------------------


def combine_P_FR_and_V_M(P_FR, V_M):
    """
    Specifically, combine the exact same PC-projection matrix V_M
    obtained in exercise 2c with the projection onto the FR
    plane found in 5b to obtain a 2 × N projection matrix.
    """

    return P_FR @ V_M.T


proj_matrix = combine_P_FR_and_V_M(P_FR, V_M)


# Now we apply the projections obtained for the interval [-150ms, 300ms] to the
# interval [-800ms, -150ms], which we will refer to as the pre-movement period

# Reload data
X, times = load_data()

# Limit the PSTH to time interval [-800ms, -150ms]
# Shape of X is (N x CT) = (182 x 7128) and T = 66
X, times, T = limit_psth(X, times, lower=-800, upper=-150)


# Directly project the N dimensional trajectories during pre-movement period
# onto the FR plane
P_rot_premov = proj_matrix @ X
print(f'{P_rot_premov.shape=}')
P_rot_premov = P_rot_premov.reshape(P_rot_premov.shape[0], -1, T)


def plt_premov_and_mov_rot_trajectories(P_rot, P_rot_premov, omega, savefig=False):
    """Plot superimposed trajectories from Q5c and Q6.

    Parameters:
    P_rot -> contains the monkey movement period trajectories from [-150ms, 200ms]
             projected on the fastest rotation (FR) plane. 
             shape is (2 x C x 36)

    P_rot -> contains the monkey pre-movement period trajectories from [-800ms, -150ms]
             projected on the same FR plane as P_rot. 
             shape is (2 x C x 66)
    """

    # Extract the first and second row of each matrix
    r1, r2 = P_rot[0], P_rot[1]
    r1_premov, r2_premov = P_rot_premov[0], P_rot_premov[1]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # --------- PLOT TRAJECTORIES FROM MONKEY MOVEMENT PERIOD -------------

    # Get the colors in which we plot the trajectories of different conditions
    # xs and ys are coordinates of initial point of trajectories.
    xs, ys = r1[:, 0], r2[:, 0]
    colors = cond_color.get_colors(xs, ys, alt_colors=False)

    # Plot all trajectories from various conditions on same plot
    for c in range(r1.shape[0]):
        ax.plot(r1[c, :], r2[c, :], color=colors[c], alpha=0.3)

    # Plot round markers on the starting point of trajectories
    cond_color.plot_start(xs, ys, colors, markersize=200)

    # Plot diamond-shaped markers on the ending point of trajectories
    xs, ys = r1[:, -1], r2[:, -1]
    cond_color.plot_end(xs, ys, colors, markersize=30)

    # ------- PLOT TRAJECTORIES FROM MONKEY PRE-MOVEMENT PERIOD -----------

    # Get the colors in which we plot the trajectories of different conditions
    # xs and ys are coordinates of FINAL point of trajectories.
    xs, ys = r1_premov[:, -1], r2_premov[:, -1]
    colors = cond_color.get_colors(xs, ys, alt_colors=True)

    # Plot all the trajectories on the same plot
    for c in range(r1_premov.shape[0]):
        ax.plot(r1_premov[c, :], r2_premov[c, :], color=colors[c])

    # Plot round markers on the starting point of trajectories
    cond_color.plot_start(
        r1_premov[:, 0], r2_premov[:, 0], colors, markersize=200)

    # Add legend
    ax.legend()

    # Add axis labels
    ax.set_xlabel('$1^{st}$ row of $P_{FR}$ matrix', fontsize='x-large')
    ax.set_ylabel('$2^{nd}$ row of $P_{FR}$ matrix', fontsize='x-large')

    # Change the tick labels size
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add title
    ax.set_title(
        f'Movement and Pre-movement trajectories on the same FR plane with $\omega$={omega:.4f}', fontsize='x-large')

    # Save the figure
    if savefig:
        save_fig(fig, 'Q6')

    plt.show()


plt_premov_and_mov_rot_trajectories(P_rot, P_rot_premov, omega, savefig=True)
