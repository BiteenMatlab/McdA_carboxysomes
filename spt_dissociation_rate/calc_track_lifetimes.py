# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:40:10 2024

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob
from lmfit import Model

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Define data name
name = 'McdA_delB_80'
# Define the integration time for imaging (in seconds)
t_int = .08
# Define the delay time between frames (in seconds)
t_delay = 0
# Define minimum track length (in frames)
min_frames = 4
# maximum gap
max_gap = 1
# Define exponential model to fit data to: 'single' or 'double'
decay_model = 'single'
# exponential fit shift (this is the lowest lifetime a track can have based on our conditions)
x0 = 0

def filepull(directory):
    """
    Finds files in directory and returns them as list

    Parameters
    ----------
    directory : string ; path to directory

    Returns
    -------
    filenames : list ; list of filename strings

    """
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]

    return filenames


def plot_single_molecule_lifetimes(lifetimes):
    """
    To calculate the 1 - CPD of the trajectory lifetimes

    Parameters
    ----------
    lifetimes : list or array ; single-molecule on times in seconds

    Returns
    -------
    x : array ; time (seconds)
    p : array ; 1 - CPD of single molecule on times in seconds

    """
    
    x, counts = np.unique(lifetimes, return_counts=True)
    cusum = np.cumsum(counts)  
    p =  1 - (cusum / cusum[-1])
    
    return (x, p)

def plot_raw_survival(lifetimes, time_steps):
    """
    To calculate the raw survival curve for single molecule on times, without
    normalization.

    Parameters
    ----------
    lifetimes : array ; single molecule on times in seconds
    time_steps : array ; unique single molecule on time values

    Returns
    -------
    x_values : array ; time (seconds)
    y_values : array ; survival curve (counts)
    
    """
   
    # Sort lifetimes
    sorted_lifetimes = np.sort(lifetimes)

    # Initialize lists for plotting
    x_values = []
    y_values = []

    for i in time_steps:
        x_values.append(i)
        survival_count = np.sum(sorted_lifetimes >= i)  # Raw count of lifetimes greater than time i
        y_values.append(survival_count)
        
    return (x_values, y_values)
    


def single_exponential_model(x, A, decay_constant):
    """
    Parameters
    ----------
    x : array ; single molecule on lifetimes (seconds)
    A : float ; amplitude scaling value for exponential function. Fit parameter.
    decay_constant : float ; decay constant of exponential function. Fit parameter. 

    Returns
    -------
    single exponential decay function

    """
    return A*np.exp(-decay_constant*(x-x0))

def biexponential_model(x, amp, decay_constant1, decay_constant2):
    """
    Parameters
    ----------
    x : array ; single molecule on lifetimes (seconds)
    A : float ; amplitude scaling value for exponential function. Fit parameter.
    decay_constant1 : float ; decay constant 1 of exponential function. Fit parameter.
    decay_constant2 : float ; decay constant 2 of exponential function. Fit parameter.

    Returns
    -------
    double exponential decay function

    """
    
    return amp * np.exp(-decay_constant1*(x-x0)) + (1-amp) * np.exp(-decay_constant2*(x-x0))

def fit_exponential(x_data, y_data, fit_model = 'single'):
    """
    Fit survival probability data to a single exponential decay model

    Parameters
    ----------
    x_data : array ; Array of x values (e.g., time).
    y_data : array ; Array of y values (survival probability).
    fit_model : string, optional ; The default is 'single'. To select which 
    model to fit to. 

    Returns
    -------
    result : Fitting result object from lmfit.
       

    """
    
    # Create a model for the single exponential
    if fit_model == 'single':
        model = Model(single_exponential_model)
        # Set up initial parameter values
        params = model.make_params(A=1000, decay_constant=1)
        
        # Perform the fit
        result = model.fit(y_data, params, x=x_data)
        
        print("decay constant: ", result.params['decay_constant'].value)
        
        mean_fit_value = result.params['decay_constant'].value
        std_fit_value = result.params['decay_constant'].stderr
        print(result.fit_report())
        
        print(mean_fit_value, std_fit_value)
        
        # calculate half life for each component
        
    elif fit_model == 'double':
        model = Model(biexponential_model)
        # Set up initial parameter values
        params = model.make_params(amp=.5, decay_constant1=0.1, decay_constant2=1)
        
        # Perform the fit
        result = model.fit(y_data, params, x=x_data)
        
        print(result.params['decay_constant1'].value, result.params['decay_constant2'].value)
        

    return result

def has_gap(values):
    """
    Determine if trajectory has frame larger than set limit

    Parameters
    ----------
    values : array ; frame numbers associated with a trajectory
    
    Returns
    -------
    bool ; if True then a gap greater than the limit was found
    
    """
    sorted_values = values
    
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > (max_gap + 1):
            return True
    return False

# load files
directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'tracks.csv' in file]

# Initialize a list to store the track lengths from all files
all_track_lengths = []

# iterate through each file
for file in track_files[:]:
    
    df = pd.read_csv(file, header=0)
    track_lengths = []
    
    for track_id in df["TRACK_N"].unique()[:]: # iterate through each track
        track_data = df[df["TRACK_N"] == track_id]
        
        gap = has_gap(track_data["FRAME_N"].tolist()) # determine if contains gap too big
        
        track_length_frame = len(track_data['FRAME_N']) # determine track length
       
        if track_length_frame >= min_frames and gap == False:
            n = (track_data['FRAME_N'].iloc[-1] - track_data['FRAME_N'].iloc[0]) + 1
            track_length = n * (t_int + t_delay)
            track_lengths.append(track_length)
            
    # Append the track lengths to the list
    all_track_lengths.extend(track_lengths)
    
    
print(len(all_track_lengths))
# get normalized survival curves
#survival_data = plot_single_molecule_lifetimes(all_track_lengths)

# raw survival curves
timesteps, counts = np.unique(all_track_lengths, return_counts=True)
survival_data = plot_raw_survival(all_track_lengths,timesteps)

# perform fit
fit_result = fit_exponential(survival_data[0], survival_data[1], fit_model=decay_model)

survival_df = pd.DataFrame({'Lifetime': survival_data[0], 'Survival Probability': survival_data[1], 
                           'Fit': fit_result.best_fit})
print(survival_df)
survival_df.to_csv(directory[:-5] + name + '_survival_probability_curve.csv',index = False)

# plot data

fig, axes = plt.subplots(figsize=(3.5,3), dpi=300)
# Plot the data and fit
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12

plt.plot(survival_data[0], survival_data[1], label='Data', linestyle = 'none', marker='o', markersize=3)
plt.plot(survival_data[0], fit_result.best_fit, 'k--', label='best fit')
#fit_result.plot_fit()
print(fit_result)
#plt.yscale('log')
#plt.xscale('log')
plt.title('Single Exponential Fit to Survival Probability')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.ylim(-100, 5000)
plt.xlim(-0.1, 15)
plt.xticks([0, 2, 4, 6, 8, 10, 12, 14])
fig.tight_layout()
plt.savefig(directory[:-5] + '20220926_Fig_Dapp_all.svg', dpi=300) 
plt.show()

# plot residuals
residuals = fit_result.residual
plt.plot(survival_data[0], residuals, label='Residuals', linestyle='--')
plt.title('Exponential Fit and Residuals')
plt.xlabel('Time (s)')
plt.ylabel('Survival Probability / Residuals')
plt.legend()
plt.show()
   