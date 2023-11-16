from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from modules import load_data, save_fig

# Load data
X, times = load_data()

# Generate a timestamp to make a unique filename
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

# Define the color map (from dark blue to red)
cmap = plt.get_cmap('coolwarm')

# Create first figure and axis
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Choose how many neurons you want to plot and under which condition
n = 51
c = 0

# Create the heatmap
heatmap = ax1.imshow(X[:n, c, :], cmap=cmap, aspect='auto', origin='lower')

# Set the x and y-axis labels
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Neurons')

# Set the x-axis and y-axis tick labels using the times list
ax1.set_xticks(np.arange(0, len(times), 10))
ax1.set_xticklabels(times[::10], rotation=45)

neuron_labels1 = [str(i) if (i % 10 == 0 and i != 0) else '' for i in range(n)]
ax1.set_yticks(range(n))
ax1.set_yticklabels(neuron_labels1)

# Set title
ax1.set_title(f'Average Firing Rates for 50 neurons in condition #{c+1}')

# Add a color bar to indicate the values
cbar = plt.colorbar(heatmap, ax=ax1)
cbar.set_label('Average Firing Rate (Hz)')

# Save the first plot to an image file
# save_fig(fig1, 'Q1_50_neurons_second_cond')

# Create second figure and axis
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Plot the first neuron activity
i = 0
ax2.plot(times, X[0, 0, :], label='Neuron #1, condition #1',
         color='royalblue')
ax2.plot(times, X[13, 0, :],
         label='Neuron #14, condition #1', color='red')
ax2.plot(times, X[0, 1, :], label='Neuron #1, condition #2',
         color='lightseagreen')
ax2.plot(times, X[13, 1, :],
         label='Neuron #14, condition #2', color='coral')

# Set the x and y axis labels and title
ax2.set_xlabel('Time (ms)')
ax2.set_ylabel('Firing rate (in Hz or spikes per second)')
ax2.set_title(f'Two different neurons in two conditions')

# Set the x axis labels
ax2.set_xlim(times[0] - 20, times[-1] + 20)
ax2.set_xticks(np.squeeze(times[::20]))
ax2.set_xticklabels(np.squeeze(times[::20]))

ax2.legend()

# save_fig(fig2, f'Q1_neuron_#{i+1}_in_cond_#{c+1}')


# Task: Plot the population average firing rate as a function of time,
# obtained by taking the average of the PSTHs across neurons and conditions.

# Create third figure and axis
fig3, ax3 = plt.subplots(figsize=(8, 6))

# Plot the population average firing rate
ax3.plot(times, X.mean(axis=(0, 1)))

# Set the x-axis tick labels
ax3.set_xlim(times[0] - 20, times[-1] + 20)
ax3.set_xticks(np.squeeze(times[::20]))
ax3.set_xticklabels(np.squeeze(times[::20]))

# Label the x-axis and y-axis
ax3.set_xlabel('Time (ms)')
ax3.set_ylabel('Average firing rate (Hz)')

# save_fig(fig3, 'Q1_avg_FR_across_neurons_and_conds')

# Show the plot
plt.show()
