# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:19:26 2023

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 13:46:53 2023

@author: p.nazarirobati
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:09:44 2023

@author: p.nazarirobati
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import quantities as pq
from elephant.gpfa import GPFA
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from elephant.conversion import BinnedSpikeTrain
from scipy import stats
import pandas as pd
from scipy.stats import chi2
from matplotlib import patches
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec


def find_bin_index(time):
    bins = np.arange(0, 2, 0.02)  # Create bins from 1 second before to 1 second after
    return np.digitize(time, bins) - 1  # Subtract 

def ellipsoid(trajectories, N):
    onset_times = []

    for i in range(len(trajectories)):
        
        onset_times.append(np.array(trajectories[i][0:3,N[i]]))
    
    onset_times = np.array(onset_times)
    print(np.shape(onset_times))
    mean_location = np.mean(onset_times, axis=0)
    covariance_matrix = np.cov(onset_times, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    confidence_level = 0.9
    degrees_of_freedom = 3  # For 3D data
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
#################################################

bin_size = .02 * pq.s
latent_dimensionality = 7

num_time_points = 100
RatName= 'Rat'

#day_id2 = '2015-04-24'
days_idx = ['2011-04-14', '2011-04-15', '2011-04-16', '2011-04-17', '2011-04-18', '2011-04-19', '2011-04-20', '2011-04-21', '2011-04-22', '2011-04-23', '2011-04-24', '2011-04-25', '2011-04-26', '2011-04-27', '2011-04-28']
#days_idx = ['2012-07-30', '2012-07-31', '2012-08-01', '2012-08-02', '2012-08-03', '2012-08-04', '2012-08-05', '2012-08-06', '2012-08-07', '2012-08-08']
#days_idx = ['2015-10-14', '2015-10-15', '2015-10-16', '2015-10-17', '2015-10-18', '2015-10-19', '2015-10-20', '2015-10-21', '2015-10-22', '2015-10-23', '2015-10-24', '2015-10-25', '2015-10-26', '2015-10-27', '2015-10-28', '2015-10-29', '2015-10-30', '2015-10-31', '2015-11-01', '2015-11-02']
#days_idx = ['2016-04-20', '2016-04-21', '2016-04-22', '2016-04-23', '2016-04-24', '2016-04-25', '2016-04-26', '2016-04-27', '2016-04-28', '2016-04-29', '2016-05-01', '2016-05-02', '2016-05-03', '2016-05-04', '2016-05-05']

fig, axs = plt.subplots(1, 7, figsize=(18, 2))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.05])  # Adjust width_ratios for subplots and colorbar

axs = axs.flatten()
num_colors = len(days_idx)
grayscale_colors = [(i/num_colors, i/num_colors, i/num_colors) for i in range(num_colors)]
for idx, day_id in enumerate(days_idx):
    filename =  r'E:\Independent Study_GPFA\gpfa_results\gpfa_dim7\resutls_rr5_' + str(day_id)+'_gpfa.pkl'
    with open(filename, 'rb') as f:
        data = pickle.load(f)


    gpfa_result = data['fit']
    gpfa_transform = data['transform']


    trajectories = gpfa_transform['latent_variable_orth']

    cov = np.mean(gpfa_transform['Vsm'][0], axis=2)
    estimated_params = gpfa_result.params_estimated




    j=0
    for i in range(latent_dimensionality):
        axs[i].set_facecolor('#FFB366')
    #row = i // 4  # Calculate row index
    #col = i % 4    # Calculate column index
        trajectory_axis =  np.array([trajectories[xx][i] for xx in range(len(trajectories))])
        trajectory_axis_avg = np.mean(trajectory_axis, axis=0)
        axs[i].plot(np.linspace(-1,1,num_time_points), trajectory_axis_avg, color=grayscale_colors[idx])  # Plot each time series
        axs[i].set_title('Factor ' + str(i+1), fontsize = 10)
    #axs[i].set_xlabel('Time from reach onset(sec)')
        axs[i].set_ylim([-1,2])
        axs[i].axvline(x=0, c='darkmagenta', linestyle='--', linewidth=.9, dashes=(2,1))
        if i>0:
            axs[i].tick_params(axis='y', which='both', left=True, labelleft=False)  # Hide y-axis ticks for other subplots

        
        
sm = ScalarMappable(cmap='gray', norm=plt.Normalize(0, 1))
sm.set_array([])  # Dummy array for color mapping

# Add a colorbar to the right side of the figure
cax = fig.add_subplot(gs[0, -1])
cax.set_position([0.9039, 0.15, 0.01, 0.7])
cbar = plt.colorbar(sm, cax=cax, orientation='vertical', shrink=0.1)
#cbar.set_label('Day', labelpad=-1)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['1', str(len(days_idx))], fontsize=8)
cbar.ax.tick_params(labelsize=5, length=2, direction='in')

plt.text(1.3, 0.45, 'Day', rotation=90, fontsize=6)
#ax.yaxis.grid(True, linestyle='-', linewidth=0.5)  # Adjust linestyle and linewidth as needed

#fig.constrained_layout = True
# Adjust layout to prevent overlapping titles
plt.tight_layout()
axs[3].set_xlabel('Time from reach onset (sec)')
# Show the plot
plt.show()


x=g55
for i in range(latent_dimensionality):
    trajectory_axis =  np.array([trajectories[xx][dim] for xx in range(len(trajectories))])
    trajectory_axis_avg = np.mean(trajectory_axis, axis=0)
    ax2[dim].plot(np.linspace(-1,1,num_time_points), trajectory_axis_avg, color='blue')
    ax2[dim].set_title('Factor ' + str(dim+1) + ' Average Activity', fontsize = 8)
    ax2[dim].set_xlabel('Time from reach onset(sec)')
        
    x_left, x_right = ax2[dim].get_xlim()
    y_low, y_high = ax2[dim].get_ylim()
    ax2[dim].set_aspect(abs((x_right-x_left)/(y_low-y_high)))


ax2[0].set_ylabel('Average Activity')
    
plt.tight_layout()
plt.show()














x=g555



data_extracted = data['spt']

binned_spt = []
for i in range(len(data_extracted)):
    bst = BinnedSpikeTrain(data_extracted[i], bin_size=0.02 * pq.s, t_start=0.0* pq.s, t_stop=2.0 *pq.s)
    bst=bst.to_array()
    binned_spt.append(bst)
    

binned_spt = np.array(binned_spt)

binned_spt_rs = np.zeros((binned_spt.shape[1], binned_spt.shape[0]*binned_spt.shape[2]))

binned_spt_rs = np.reshape(binned_spt, np.shape(binned_spt_rs))

binned_spt_rs = stats.zscore(binned_spt_rs, axis=1)
binned_spt_ls = [binned_spt_rs[xx,:] for xx in range(binned_spt_rs.shape[0])]


gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)
gpfa_2dim.fit(data_extracted)

gpfa_2dim_transform = gpfa_2dim.transform(data_extracted, returned_data=['latent_variable_orth', 'Vsm', 'y'])

cov = gpfa_2dim_transform['Vsm']

trajectories = gpfa_2dim_transform['latent_variable_orth']
score = gpfa_2dim.score(data_extracted)
estimated_params = gpfa_2dim.params_estimated


#x=g55

fig = plt.figure()
ax = plt.axes(projection='3d')

linewidth_single_trial = 0.5
color_single_trial = 'C0'
alpha_single_trial = 0.5

linewidth_trial_average = 2
color_trial_average = 'C1'

ax.set_title('Latent dynamics extracted by GPFA')
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')

ax.set_aspect('auto')
# single trial trajectories
for single_trial_trajectory in trajectories:
    ax.plot3D(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2], '-', lw=linewidth_single_trial, c=color_single_trial, alpha=alpha_single_trial)
# trial averaged trajectory
average_trajectory = np.mean(trajectories, axis=0)
ax.plot3D(average_trajectory[0], average_trajectory[1], average_trajectory[2], '-', lw=linewidth_trial_average, c=color_trial_average, label='Trial averaged trajectory')
ax.legend()

plt.tight_layout()
plt.show()

fig, ax2 = plt.subplots(1, latent_dimensionality, figsize = (10,4))

for dim in range(latent_dimensionality):
    trajectory_axis =  np.array([trajectories[xx][dim] for xx in range(len(trajectories))])
    trajectory_axis_avg = np.mean(trajectory_axis, axis=0)
    ax2[dim].plot(np.linspace(-1,1,num_time_points), trajectory_axis_avg, color='blue')
    ax2[dim].set_title('Factor ' + str(dim+1) + ' Average Activity', fontsize = 8)
    ax2[dim].set_xlabel('Time from reach onset(sec)')
        
    x_left, x_right = ax2[dim].get_xlim()
    y_low, y_high = ax2[dim].get_ylim()
    ax2[dim].set_aspect(abs((x_right-x_left)/(y_low-y_high)))


ax2[0].set_ylabel('Average Activity')
    
plt.tight_layout()
plt.show()

#### Cross-Validation to find optimal model dimension
x=g5555
x_dims = [1, 2, 3, 4, 5]
log_likelihoods = []
for x_dim in x_dims:
    gpfa_cv = GPFA(bin_size=bin_size, x_dim=x_dim)
    #cv = LeaveOneOut()
    cv = 5

    # estimate the log-likelihood for the given dimensionality as the mean of the log-likelihoods from 3 cross-vailidation folds
    #cv_log_likelihoods = cross_val_score(gpfa_cv, data_extracted, cv=3, n_jobs=3, verbose=True)
    cv_loo = cross_val_score(gpfa_cv, data_extracted, cv=cv, n_jobs=-1, verbose=True)
    log_likelihoods.append(np.mean(cv_loo))
    
    
f = plt.figure(figsize=(7, 5))
plt.xlabel('Dimensionality of latent variables')
plt.ylabel('Log-likelihood')
plt.plot(x_dims, log_likelihoods, '.-')
plt.plot(x_dims[np.argmax(log_likelihoods)], np.max(log_likelihoods), 'x', markersize=10, color='r')
plt.tight_layout()
plt.show()    

x=g666
gpfa.fit(data_extracted) 

results = gpfa.transform(data_extracted, returned_data=['latent_variable_orth','latent_variable']) 

latent_variable_orth = results['latent_variable_orth']  

latent_variable = results['latent_variable']   