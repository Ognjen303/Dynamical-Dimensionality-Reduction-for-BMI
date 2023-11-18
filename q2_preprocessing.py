import numpy as np
import matplotlib.pyplot as plt
from modules import load_data, save_fig

X, times = load_data()


def compute_max(X):
    """Compute the maximum of each neuron across conditions and time."""
    return np.max(X, axis=(1, 2))


def plot_max_scatter(X):
    """Plot a scatter plot of the neurons maximum
    (across conditions and time) firing rates."""

    fig1, ax1 = plt.subplots(figsize=(8, 6))

    max_X = compute_max(X)

    ax1.scatter(range(X.shape[0]), max_X)
    ax1.set_title(
        'Plot of maximum firing rate (in Hz) of each neuron across conditions and time')
    ax1.set_xlabel('Neuron')

    plt.show()


def plot_histogram(X, n_bins=15):
    """Plot a histogram of the neurons' maximum
    (across conditions and time) firing rates.
    """

    # Figure and axis
    fig2, ax2 = plt.subplots(figsize=(7, 6))

    max_X = compute_max(X)

    # Plot histogram
    ax2.hist(max_X, bins=n_bins)

    # Change the tick labels size
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)

    # Set x-axis and y-axis labels and title
    ax2.set_xlabel('Maximum firing rate (Hz)', fontsize='x-large')
    ax2.set_ylabel('Neuron count', fontsize='x-large')
    ax2.set_title(
        'Histogram of pre-processed max firing rates of neurons across conditions & time. 15 bins.', fontsize='medium')

    # save_fig(fig2, filename='Q2_preprocesed_Hist')

    # plt.show()
    plt.clf()


plot_histogram(X)


def normalise(X):
    """
    Separately for each neuron normalize its PSTH according to:
    psth = (psth - b) / (a - b + 5)
    where psth is the PSTH of the neuron in all conditions, and a and b are, respectively, the
    maximum and minimum value of this neuron's PSTH across both times and conditions. This step
    ensures that the normalized activities of different neurons have the same scale and approximate
    range of variations. Henceforth (unless otherwise stated) we will work with this mean-centered
    and normalized PSTH array, which we keep denoting by X.

    input:
    X -> numpy array of dimensions (N x C x T), where
    N = 182 neurons, C = 108 various conditions and T = 130 time bins
    """

    a = np.max(X, axis=(1, 2)).reshape(X.shape[0], 1, 1)
    b = np.min(X, axis=(1, 2)).reshape(X.shape[0], 1, 1)
    return 1.0 * (X - b) / (a - b + 5)


def mean_centering(X):
    """
    Remove from X its cross-condition mean (calculated and subtracted
    separately for each time bin and neuron).
    """

    return X - np.mean(X, axis=1).reshape(X.shape[0], 1, X.shape[2])

# --------------------- DIMENSIONALITY REDUCITON BY PCA --------------------

# From this step until exercise 6 we will only work
# with the PSTHs limited to the interval from −150ms to +300ms relative to movement onset (we
# will however keep using X to denote the corresponding slice of the normalized and mean-removed
# PSTH array, and use T to denote the number of time bins, now equal to 46 in this interval)


def limit_psth(X, times, lower=-150, upper=300):
    """Limits the psth to interval
    from -150ms to +300ms relative to movement onset.

    Input:
    X -> data matrix with shape (N x C x T), where T = 130.
    times -> ndarray with shape (T x 1), where T = 130.

    Output:
    X -> data matrix with shape (N x CT), where T = 46.
    times -> ndarray with shape (T x 1), where T = 46.
    T -> new number of time bins equal to 46.
    """

    X = mean_centering(normalise(X))

    # Create a boolean mask to select values within the specified range
    # mask shape is (130, 1)
    mask = (times >= lower) & (times <= upper)
    mask = mask[:, 0]  # mask shape is now (130,)

    times = times[mask]

    # number of time bins is now 46
    T = times.shape[0]

    X = X[:, :, mask]  # shape of X is now (N, C, 46)
    X = X.reshape(X.shape[0], -1)  # shape of X is now (N, Cx46)

    return X, times, T


# Just for histogram plotting purposes in Q2 after preprocessing

# X, times, T = limit_psth(X, times, lower=-800, upper=500)

# print(f'{X.shape=}')
# X = X.reshape(X.shape[0], -1, T)
# plot_histogram(X)


X, times, T = limit_psth(X, times)

# PCA


def pca_proj_matrix(X, M=12):
    """
    Find the eigenvectors and eigenvalues of S_hat = 1/T * X @ X.T
    and take the top M=12 principle components in the neuron activity space.

    input:
    X -> matrix of shape (N x CT) (T should be 46)
    M -> # Number of principal components to select

    output:
    V_M -> matrix of shape (N x M)
    """

    S_hat = 1/T * X @ X.T
    _, evecs = np.linalg.eig(S_hat)

    # Select the top M eigenvectors
    V_M = evecs[:, :M]

    return V_M


def pca_dim_reduction(X, M=12):
    """
    Projecting onto the first M = 12 principle
    components in the neuron activity space.

    Output:
    Z -> matrix of projected neuron activity
         with shape M × CT = 12 × 4968. We denote with
         Z[i, n] the elements of Z.

    """

    V_M = pca_proj_matrix(X, M)
    Z = V_M.T @ X
    del V_M

    return Z


V_M = pca_proj_matrix(X)
Z = pca_dim_reduction(X)

# Reshape Z back into a tensor of shape (M x C x T)
Z = Z.reshape(12, -1, T)

dZ = Z[:, :, 1:] - Z[:, :, :-1]

# shape is (M x C(T-1))
dZ = dZ.reshape(12, -1)

# shape is (M x C x (T-1))
Z = Z[:, :, :-1]

# shape is (M x C(T-1))
Z = Z.reshape(12, -1)

# unconstrained MLE estimate of A

A_MLE_un = dZ @ Z.T @ np.linalg.inv(Z @ Z.T)


fig, ax = plt.subplots(figsize=(10, 8))
pos = ax.imshow(A_MLE_un, cmap='coolwarm', interpolation='none')
fig.colorbar(pos, ax=ax)

ax.set_title('Values of unconstrained ML estimate of A', fontsize='large')
# Change the tick labels size
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

save_fig(fig, 'A_MLE_uncon')

print(f'{A_MLE_un=}')
print(f'{A_MLE_un.shape=}')



plt.show()


print(V_M.shape)
print(Z.shape)

print(np.dot(Z[:, :-1], Z[:, :-1].T).shape)
