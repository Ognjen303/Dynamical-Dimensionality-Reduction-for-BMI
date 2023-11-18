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


def plot_pc1_pc2_plane(Z, plt_title=None, savefig=True, T=46, **kwargs):
    """
    Plot of trajectories in the PC1-PC2 plane (corresponding
    to 0 and 1 left-indices of Z, which for plotting you would
    reshape back into a 3D array). Superimpose
    the trajectories for all conditions in the same plot.

    Input:
    Z -> shape (M x CT)
    savefig (boolean) -> True if you want to save the plot
    T (int) -> Number of timebins used in the last dimension of Z
    Q (int) -> Indicates which question we are solving.
               Q can take values 3 or 5.
    """

    Q = kwargs.get('Q', None)

    if not isinstance(plt_title, str):
        raise ValueError('Plot title must be a string')

    if not (Q == 3 or Q == 5):
        raise ValueError(
            'You can plot only in questions 3 or 5. Please indicate which.')

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

    # Change the tick labels size
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add legend
    ax.legend(prop={'size': 14})

    if Q == 3 or Q is None:  # We are solving Q3
        # Add axis labels and title for Q3
        ax.set_xlabel('1st Principle Component', fontsize='x-large')
        ax.set_ylabel('2nd Principle Component', fontsize='x-large')
        ax.set_title(plt_title, fontsize='large')

    elif Q == 5:
        # Add axis labels and title for Q5
        ax.set_xlabel('$1^{st}$ row of P matrix', fontsize='x-large')
        ax.set_ylabel('$2^{nd}$ row of P matrix', fontsize='x-large')
        ax.set_title(plt_title, fontsize='large')

    # Save the figure
    if savefig:
        if Q == 3:
            save_fig(fig, 'Q3_Trajectories_in_PC1_PC2_plane')
        elif Q == 5:
            save_fig(fig, 'Q5_P_FR_Trajectories_3rd_fastest')

    plt.show()


plt_title = 'Plot of neural activity trajectories in the PC1-PC2 plane'

# plot_pc1_pc2_plane(Z, plt_title, savefig=False, Q=3)
