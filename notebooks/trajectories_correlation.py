import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import quantities as pq
import matplotlib.pyplot as plt
from scipy.stats import chi2

"""
Created on Sat Dec  2 13:46:53 2023
@author: p.nazarirobati
"""
# Configuration Variables
BIN_SIZE = 0.02 * pq.s
LATENT_DIMENSIONALITY = 7
NUM_TIME_POINTS = 100
RAT_NAME = 'Rat'
DAYS_IDX = [
    '2016-04-20', '2016-04-21', '2016-04-22', '2016-04-23', '2016-04-24', '2016-04-25',
    '2016-04-26', '2016-04-27', '2016-04-28', '2016-04-29', '2016-05-01', '2016-05-02',
    '2016-05-03', '2016-05-04', '2016-05-05'
]

# Helper Functions
def find_bin_index(time, bins=np.arange(0, 2, 0.02)):
    """Find the bin index for a given time."""
    return np.digitize(time, bins) - 1

def compute_ellipsoid(trajectories, indices):
    """Compute ellipsoid parameters based on trajectory data."""
    onset_times = np.array([trajectories[i][0:3, indices[i]] for i in range(len(trajectories))])
    mean_location = np.mean(onset_times, axis=0)
    covariance_matrix = np.cov(onset_times, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    confidence_level = 0.9
    degrees_of_freedom = 3
    scale_factor = np.sqrt(chi2.ppf(confidence_level, degrees_of_freedom))
    scaled_eigenvalues = np.sqrt(eigenvalues) * scale_factor

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = scaled_eigenvalues[0] * np.outer(np.cos(u), np.sin(v))
    y = scaled_eigenvalues[1] * np.outer(np.sin(u), np.sin(v))
    z = scaled_eigenvalues[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x[i])):
            [x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], eigenvectors) + mean_location
    return x, y, z

# Main Processing
fig, axs = plt.subplots(1, 2, figsize=(12, 3))
trajectories_all = []
MAIN_PATH = 'E:/Independent Study_GPFA'
for day_id in DAYS_IDX:
    filename = Path(f"{MAIN_PATH}/gpfa_results/gpfa_dim7/resutls_rr9_{day_id}_gpfa.pkl")
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    trajectories = np.array(list(data['transform']['latent_variable_orth']))
    trajectories_all.append(trajectories)

# Calculate Optimal Trajectory
optimal_trajectory = [np.mean(trajectories_all[xx][:, 0:3, :], axis=0) for xx in range(-3, 0)]
optimal_trajectory = np.mean(np.array(optimal_trajectory), axis=0)

# Correlation and Distance Calculation
corr_all = []
for trajectory_day in trajectories_all:
    corr_day = []
    for j in range(trajectory_day.shape[0]):
        correlation = np.corrcoef(
            trajectory_day[j, 0:3, :].flatten(), optimal_trajectory.flatten()
        )[0, 1]
        corr_day.append(correlation)
    corr_all.append(corr_day)

avg_corr = [np.abs(np.mean(day_corr)) for day_corr in corr_all]
std_corr = [np.abs(np.std(day_corr)) for day_corr in corr_all]

# Save Standard Deviations
np.save('rr9_std_corr.npy', std_corr)

# Plot Results
axs[0].plot(np.arange(1, len(DAYS_IDX) + 1), avg_corr, color='blue')
axs[1].plot(np.arange(1, len(DAYS_IDX) + 1), std_corr, color='red')

axs[0].set_ylabel('Trajectory Correlation - Mean', fontsize=10)
axs[1].set_ylabel('Trajectory Correlation - SD', fontsize=10)
axs[0].set_xlabel('Training Day', fontsize=10)
axs[1].set_xlabel('Training Day', fontsize=10)

axs[0].set_ylim([-.1, 1])
axs[1].set_ylim([0, .3])
for ax in axs:
    ax.tick_params(axis='both', which='both', direction='in', labelsize=10, length=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.suptitle('Rat 4')
plt.tight_layout()

# Save Figures
fig_width_cm = 22
fig_width_inches = fig_width_cm / 2.54
height_inches = 5 * (fig_width_inches / fig_width_cm)
fig.set_size_inches(fig_width_inches, height_inches)

plt.savefig('trajectories_corr4TIFF.tiff', dpi=300, format='tiff', bbox_inches='tight')
plt.savefig('trajectories_corr4SVG.svg', dpi=300, format='svg', bbox_inches='tight')
plt.show()
