# Neural Latent Dynamics Analysis Using GPFA

This repository contains Python scripts for analyzing neural spike data using Gaussian Process Factor Analysis (GPFA). It provides tools to preprocess spike data, extract latent dynamics, visualize 3D trajectories, and evaluate their stability and similarity over the course of motor learning. The framework is designed to handle multi-day neural recordings and provide insights into the temporal evolution of neural latent variables.

## Features
- **Spike Data Preprocessing**: Converts raw neural spike times into trial-aligned data for GPFA analysis.
- **Latent Dynamics Extraction**: Utilizes GPFA to uncover latent neural dynamics from binned spike trains.
- **3D Trajectory Visualization**: Visualizes extracted latent dynamics as 3D trajectories and ellipsoids representing variance.
- **Cross-Validation**: Determines the optimal dimensionality for latent dynamics using log-likelihood estimation.
- **Trajectory Analysis**: Calculates correlations and stability metrics of trajectories across training days.
