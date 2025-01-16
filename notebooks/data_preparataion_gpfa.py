import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import quantities as pq
import neo

# Configuration Variables
RAT_NAME = 'rr5'
DAY_ID = '2011-04-22'
DAY_ID2 = '2015-04-22'

MAIN_PATH = 'E:/Master Project'
SPIKE_PATH = Path(f"{MAIN_PATH}/dataset/Database/{RAT_NAME}/{DAY_ID}/spikes")
REACH_PATH = Path(f"{MAIN_PATH}/outputs/{RAT_NAME}/{DAY_ID2}/epochs.treach.txt")

T_START_SHIFT = 10000  # in 0.1 ms
T_END_SHIFT = 10000    # in 0.1 ms
BIN_WIDTH = 200        # in 0.1 ms

OUTPUT_FILENAME = Path(f"{MAIN_PATH}/Analysis/gpfa/data/{RAT_NAME}_{DAY_ID}_gpfa.pkl")

# Helper Functions
def load_spike_data(spike_path):
    """Load spike data from the specified directory."""
    spike_data = []
    for file in sorted(spike_path.glob("cell*")):
        with open(file, 'r') as f:
            spikes = list(map(float, f.read().split()))
            spike_data.append([int(spike / 100) for spike in spikes])  # Convert to 0.1 ms time stamps
    return spike_data

def process_reach_data(reach_path):
    """Load and process reach data."""
    reach_info = pd.read_csv(reach_path, sep='\s+')
    reach_info_success = reach_info[reach_info['issingle'] == 1].reset_index(drop=True)
    return reach_info_success

def extract_spike_trials(spike_data, reach_info, t_start_shift, t_end_shift):
    """Extract spike trials around reaching times."""
    spike_trials = []
    total_shift = t_start_shift + t_end_shift

    for _, row in reach_info.iterrows():
        reach_onset = row['tadvance']
        trial_spikes = []

        for neuron_spikes in spike_data:
            spikes_around_reach = [
                (spike - (reach_onset - t_start_shift)) / 10000
                for spike in neuron_spikes
                if reach_onset - t_start_shift < spike < reach_onset + t_end_shift
            ]
            spike_train = neo.SpikeTrain(
                spikes_around_reach * pq.s,
                t_start=0 * pq.s,
                t_stop=total_shift / 10000 * pq.s
            )
            trial_spikes.append(spike_train)

        spike_trials.append(trial_spikes)

    return spike_trials

def save_data_to_pickle(data, filename):
    """Save data to a pickle file."""
    with open(filename, "wb") as f:
        pickle.dump(data, f)

# Main Processing
if __name__ == "__main__":
    # Load data
    spike_data = load_spike_data(SPIKE_PATH)
    reach_info_success = process_reach_data(REACH_PATH)

    # Extract spike trials
    spike_data_trials = extract_spike_trials(
        spike_data, reach_info_success, T_START_SHIFT, T_END_SHIFT
    )

    # Save results
    data_to_save = {"spt": spike_data_trials}
    save_data_to_pickle(data_to_save, OUTPUT_FILENAME)

    print(f"Data saved successfully to {OUTPUT_FILENAME}")
