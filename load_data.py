"""Module providing a function to load a .npz file."""
from numpy import load


def load_data(filename='psths.npz'):
    """Function returns:

    a 3D Numpy array, X, with shape N × C × T, 
    containing the so-called PSTH’s of N = 182 neurons in T = 130 time-bins
    and C = 108 task conditions. PSTH stands for “peristimulus time histogram".
    and refers to the sequence of average spike counts or firing rates of a neuron
    in different time bins in a time interval around the onset of a stimulus or a movement.
    In our case, the interval goes from -800 ms (milliseconds) to +500 ms relative to 
    the onset of hand movement; the interval was divided into 130 time bins of 10 ms width.
    As is commonly done, the trial-averaged spike counts have been divided
    by the bin width (in units of seconds), such that 
    X[i, c, t] is the average firing rate of neuron i in
    the t-th time bin in condition c (in units of Hz or spikes per second).

    A 1D array, times, with the start time (in milliseconds) of the different PSTH bins
    relative to movement onset (see Fig. 1A).
    """

    data = load(filename)
    return data['X'], data['times']
