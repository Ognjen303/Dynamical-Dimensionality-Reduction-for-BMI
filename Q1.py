import numpy as np
import cond_color
import matplotlib.pyplot as plt
from load_data import load_data

X, times = load_data()

print(X.shape)
print(times.shape)

fig, ax = plt.subplots()

ax.scatter(times, X[0, 0, :])
ax.set_xlabel('Time (ms)')
ax.set_ylabel(
    'Average firing rate of neuron 1 in condition 1 in spikes per second', fontsize=10)
plt.show()
