import numpy as np
import matplotlib.pyplot as plt
import cond_color
from modules import load_data, save_fig
from q2_preprocessing import limit_psth, pca_dim_reduction

# Load data
X, times = load_data()

# Limit the PSTH to time interval between -150ms and +300ms
# Shape of X is (N x CT) = (12 x 4968) and T = 46
X, times, T = limit_psth(X, times)

# Dimensionality reduction by PCA
Z = pca_dim_reduction(X)


def plot_pc1_pc2_plane(Z, savefig=True, T=46):
    """
    Plot of trajectories in the PC1-PC2 plane (corresponding
    to 0 and 1 left-indices of Z, which for plotting you would
    reshape back into a 3D array). Superimpose
    the trajectories for all conditions in the same plot.

    Input:
    Z -> shape (M x CT)
    """

    print(Z.shape)

    # Extract the principle components
    pc1, pc2 = Z[0, :], Z[1, :]

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 9))

    # Reshape pc1 and pc2 for ploting purposes
    pc1 = pc1.reshape(-1, T)
    pc2 = pc2.reshape(-1, T)

    # Get the colors in which we plot the trajectories of different conditions
    # xs and ys are coordinates of initial point of trajectories.
    xs, ys = pc1[:, 0], pc2[:, 0]
    colors = cond_color.get_colors(xs, ys, alt_colors=False)
    print(f'{len(colors)}')

    print(f'{pc1.shape=}')
    # Plot all trajectories from various conditions on same plot
    for c in range(pc1.shape[0]):
        ax.plot(pc1[c, :], pc2[c, :], color=colors[c])

    # Plot round markers on the starting point of trajectories
    cond_color.plot_start(xs, ys, colors, markersize=200)

    # Plot diamond-shaped markers on the ending point of trajectories
    xs, ys = pc1[:, -1], pc2[:, -1]
    cond_color.plot_end(xs, ys, colors, markersize=30)

    # Add legend
    ax.legend()

    # Add axis labels and title for Q3
    # ax.set_xlabel('1st Principle Component')
    # ax.set_ylabel('2nd Principle Component')
    # ax.set_title('Plot of neural activity trajectories in the PC1-PC2 plane')

    ax.set_xlabel('1st Eigenvector of P_FR')
    ax.set_ylabel('2nd Eigenvector of P_FR')
    ax.set_title(
        'Trajectories in fastest rotation plane P_FR in time range [-150ms, 200ms]')

    # Save the figure
    if savefig:
        # save_fig(fig, 'Q3_Trajectories_in_PC1_PC2_plane')
        save_fig(fig, 'Q5_P_FR_Trajectories')

    plt.show()


# plot_pc1_pc2_plane(Z, savefig=True)
