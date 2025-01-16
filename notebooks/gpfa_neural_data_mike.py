import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import quantities as pq
from elephant.gpfa import GPFA
from elephant.conversion import BinnedSpikeTrain
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Configuration Variables
BIN_SIZE = 0.02 * pq.s
LATENT_DIMENSIONALITY = 7
NUM_TIME_POINTS = 100

MAIN_PATH = 'E:/Master Project'
FILENAME = Path(f"{MAIN_PATH}/Analysis/gpfa/data/rr5_2011-04-24_gpfa.pkl")
STRUCTURED_DATA_FILENAME = Path(f"{MAIN_PATH}/structured_data.mat")

# Helper Functions
def load_pickle_file(filename):
    """Load data from a pickle file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def bin_spike_trains(spike_trains, bin_size, t_start, t_stop):
    """Convert spike trains to binned spike trains."""
    return np.array([
        BinnedSpikeTrain(st, bin_size=bin_size, t_start=t_start, t_stop=t_stop).to_array()
        for st in spike_trains
    ])

def save_mat_file(filename, data):
    """Save data to a .mat file."""
    sio.savemat(filename, {'D': data}, format='5', long_field_names=True)

def calculate_cumulative_variance(C, Corth, total_variance, latent_dimensionality):
    """Calculate cumulative variance explained by latent variables."""
    cumulative_variance = []
    for i in range(1, latent_dimensionality + 1):
        var_explained = np.trace(
            np.dot(C[:i, :], np.dot(np.cov(Corth[:, :i].T), C[:i, :]).T)
        )
        cumulative_variance.append(var_explained / total_variance)
    return cumulative_variance

def plot_latent_dynamics(trajectories):
    """Plot latent dynamics extracted by GPFA."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for single_trial_trajectory in trajectories:
        ax.plot3D(
            single_trial_trajectory[0],
            single_trial_trajectory[1],
            single_trial_trajectory[2],
            '-', lw=0.5, color='C0', alpha=0.5
        )

    average_trajectory = np.mean(trajectories, axis=0)
    ax.plot3D(
        average_trajectory[0],
        average_trajectory[1],
        average_trajectory[2],
        '-', lw=2, color='C1', label='Trial averaged trajectory'
    )
    ax.set_title('Latent dynamics extracted by GPFA')
    ax.set_xlabel('Dim 1')
    ax.set_ylabel('Dim 2')
    ax.set_zlabel('Dim 3')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_average_activity(trajectories, latent_dimensionality, num_time_points):
    """Plot average activity for each latent dimension."""
    fig, axes = plt.subplots(1, latent_dimensionality, figsize=(10, 4))

    for dim in range(latent_dimensionality):
        trajectory_axis = np.array([t[dim] for t in trajectories])
        trajectory_axis_avg = np.mean(trajectory_axis, axis=0)
        axes[dim].plot(
            np.linspace(-1, 1, num_time_points),
            trajectory_axis_avg,
            color='blue'
        )
        axes[dim].set_title(f'Factor {dim + 1} Average Activity', fontsize=8)
        axes[dim].set_xlabel('Time from reach onset (sec)')
        axes[dim].set_ylim([-1, 1])
        x_left, x_right = axes[dim].get_xlim()
        y_low, y_high = axes[dim].get_ylim()
        axes[dim].set_aspect(abs((x_right - x_left) / (y_low - y_high)))

    axes[0].set_ylabel('Average Activity')
    plt.tight_layout()
    plt.show()

def cross_validate_gpfa(data, bin_size, x_dims, cv_folds=5):
    """Perform cross-validation to determine the optimal latent dimensionality."""
    log_likelihoods = []
    for x_dim in x_dims:
        gpfa = GPFA(bin_size=bin_size, x_dim=x_dim)
        scores = cross_val_score(gpfa, data, cv=cv_folds, n_jobs=-1, verbose=True)
        log_likelihoods.append(np.mean(scores))

    plt.figure(figsize=(7, 5))
    plt.plot(x_dims, log_likelihoods, '.-', label='Log-likelihood')
    plt.xlabel('Dimensionality of latent variables')
    plt.ylabel('Log-likelihood')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return log_likelihoods

# Main Processing
if __name__ == "__main__":
    # Load data
    data = load_pickle_file(FILENAME)
    data_extracted = data['spt']

    # Bin spike trains
    binned_spt = bin_spike_trains(data_extracted, BIN_SIZE, 0.0 * pq.s, 2.0 * pq.s)

    # Create structured data for MATLAB
    data_structures = [{'data': trial} for trial in binned_spt]
    save_mat_file(STRUCTURED_DATA_FILENAME, data_structures)

    # Reshape binned data
    binned_spt_rs = binned_spt.reshape(
        binned_spt.shape[1], -1
    )

    # Train GPFA model
    gpfa = GPFA(bin_size=BIN_SIZE, x_dim=LATENT_DIMENSIONALITY)
    gpfa.fit(data_extracted)

    # Transform data
    gpfa_results = gpfa.transform(
        data_extracted,
        returned_data=['latent_variable_orth', 'Vsm', 'VsmGP', 'y']
    )

    trajectories = gpfa_results['latent_variable_orth']
    estimated_params = gpfa.params_estimated

    # Calculate cumulative variance
    C = estimated_params['C'].T
    Corth = estimated_params['Corth']
    total_variance = np.trace(np.cov(binned_spt_rs))
    cumulative_variance = calculate_cumulative_variance(
        C, Corth, total_variance, LATENT_DIMENSIONALITY
    )

    # Plot results
    plot_latent_dynamics(trajectories)
    plot_average_activity(trajectories, LATENT_DIMENSIONALITY, NUM_TIME_POINTS)

    # Cross-validation
    x_dims = [1, 2, 3, 4, 5]
    cross_validate_gpfa(data_extracted, BIN_SIZE, x_dims)
