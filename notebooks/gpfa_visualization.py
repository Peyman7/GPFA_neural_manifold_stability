import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import quantities as pq
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2
from elephant.gpfa import GPFA
from elephant.conversion import BinnedSpikeTrain
from sklearn.model_selection import cross_val_score

# Configuration Variables
BIN_SIZE = 0.02 * pq.s
LATENT_DIMENSIONALITY = 7
NUM_TIME_POINTS = 100
RAT_NAME = 'Rat4'
DAY_ID = '2016-05-05'
DAY_ID_X = 15

GPFA_FILENAME = Path(f"E:/Independent Study_GPFA/gpfa_results/gpfa_dim7/resutls_rr9_{DAY_ID}_gpfa.pkl")
REACH_FILES = Path(f"E:/Master Project/outputs/rr9/{DAY_ID}/epochs.treach.txt")

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

# Load Data
reach_info = pd.read_csv(REACH_FILES, sep='\s+')
reach_info = reach_info[reach_info['issingle'] == 1].reset_index()
reach_info = reach_info.iloc[:, 4:7] / 10000
reach_info['tadvance'] -= 1
reach_info['GraspBinIndex'] = reach_info.apply(lambda row: find_bin_index(row['tgrasp'] - row['tadvance']), axis=1)

with open(GPFA_FILENAME, 'rb') as f:
    gpfa_data = pickle.load(f)

gpfa_result = gpfa_data['fit']
gpfa_transform = gpfa_data['transform']
trajectories = gpfa_transform['latent_variable_orth']

# Plot Trajectories and Ellipsoid
fig = plt.figure(figsize=(6.5, 4))
ax = plt.axes(projection='3d')
ax.set_title(f'{RAT_NAME} - Day {DAY_ID_X}', fontsize=11)
ax.set_xlabel('Factor 1', fontsize=8)
ax.set_ylabel('Factor 2', fontsize=8)
ax.set_zlabel('Factor 3', fontsize=8)
ax.set_aspect('auto')

N = list(reach_info['GraspBinIndex'])
x, y, z = compute_ellipsoid(trajectories, N)
x2, y2, z2 = compute_ellipsoid(trajectories, [50] * len(reach_info))

mean_onset = np.mean(np.array([trajectories[i][0:3, 50] for i in range(len(trajectories))]), axis=0)
mean_grasp = np.mean(np.array([trajectories[i][0:3, N[i]] for i in range(len(trajectories))]), axis=0)

for j, single_trial_trajectory in enumerate(trajectories):
    ax.plot3D(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2], '-', lw=0.5, c='black', alpha=0.5)
    ax.scatter(single_trial_trajectory[0][N[j]], single_trial_trajectory[1][N[j]], single_trial_trajectory[2][N[j]], s=2, color='lightpink')
    ax.scatter(single_trial_trajectory[0][50], single_trial_trajectory[1][50], single_trial_trajectory[2][50], s=2, color='lightgreen')
    ax.plot(x[0, :], y[0, :], z[0, :], color='magenta')
    ax.plot(x2[0, :], y2[0, :], z2[0, :], color='green')

ax.scatter(mean_onset[0], mean_onset[1], mean_onset[2], s=8, c='green')
ax.scatter(mean_grasp[0], mean_grasp[1], mean_grasp[2], s=8, c='darkmagenta')

average_trajectory = np.mean(trajectories, axis=0)
ax.plot3D(average_trajectory[0], average_trajectory[1], average_trajectory[2], '-', lw=2, c='C1', label='Trial averaged trajectory')
plt.tight_layout()
plt.show()

# Save Figures
fig_width_cm = 5.8
fig_width_inches = fig_width_cm / 2.54
height_inches = 4.3 / 2.54
fig.set_size_inches(fig_width_inches, height_inches)
plt.savefig('gpfa_rr5_d12.eps', dpi=600, format='eps', bbox_inches='tight')
plt.savefig('gpfa_rr5_d12.tiff', dpi=600, format='tiff', bbox_inches='tight')
plt.savefig('gpfa_rr5_d12.jpeg', dpi=600, format='jpeg', bbox_inches='tight')

# Cross-Validation
x_dims = [1, 2, 3, 4, 5]
log_likelihoods = []
for x_dim in x_dims:
    gpfa_cv = GPFA(bin_size=BIN_SIZE, x_dim=x_dim)
    scores = cross_val_score(gpfa_cv, gpfa_data['spt'], cv=5, n_jobs=-1, verbose=True)
    log_likelihoods.append(np.mean(scores))

plt.figure(figsize=(7, 5))
plt.plot(x_dims, log_likelihoods, '.-', label='Log-likelihood')
plt.xlabel('Dimensionality of latent variables')
plt.ylabel('Log-likelihood')
plt.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), 'x', markersize=10, color='r')
plt.tight_layout()
plt.show()
