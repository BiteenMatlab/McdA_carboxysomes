# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:07:37 2024

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

# Define data name or label
name = 'Hn_2'
# Define the integration time for imaging (in seconds)
t_int = 1
# Define the time delay between frames in seconds
t_delay = 0
# Define minimum track length (in frames)
min_frames = 25
# Define the pixel size in microns
pixel_size = 0.066

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def calculate_velocity(trajectory, time):
    velocities = np.diff(trajectory, axis=0) / np.diff(time)[:, None]
    print(velocities)
    return velocities


def calculate_vacf(velocities, max_lag):
    n = len(velocities)
    vacf = np.zeros(max_lag + 1)
    
    for lag in range(max_lag + 1):
        dot_products = [
            np.dot(velocities[i], velocities[i + lag]) 
            for i in range(n - lag)
        ]
        vacf[lag] = np.mean(dot_products)
        
    return vacf

def calc_vacf(velocities):
   
    # Number of time steps
    n_steps = len(velocities)
    # Compute the velocity autocorrelation function
    vacf = np.zeros(n_steps)
    for t in range(n_steps):
        autocorr = 0
        for i in range(n_steps - t):
            autocorr += np.dot(velocities[i], velocities[i + t])
        vacf[t] = autocorr / (n_steps - t)

    # Normalize VACF: vacf(t) / vacf(0)
    vacf /= vacf[0]

directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'spots' in file]
print(track_files)
# Initialize a list to store the track lengths from all files

all_track_vacf_out = []
track_ids = []

# load through each file
for file in track_files[:]:
    
    # read csv as dataframe
    df = pd.read_csv(file, header=0)
    # Calculate the radius of confinement
    vacf_out = []
    # loop through each track
    for track_id in df["TRACK_ID"].unique():
        # load track data
        track_data = df[df["TRACK_ID"] == track_id]
        n = len(track_data)
        
        # calculate radius of confinement if track is min_frames long
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['POSITION_X'].to_numpy()[:min_frames]*pixel_size
            track_y_coords = track_data['POSITION_Y'].to_numpy()[:min_frames]*pixel_size
            time = track_data['FRAME'].to_numpy()[:min_frames]
            #print(len(track_x_coords), len(track_y_coords))
                
            track = np.vstack((track_x_coords, track_y_coords)).T
            velocities = calculate_velocity(track, time)
            mean_velocity = np.mean(velocities)
            velocities_shifted = velocities - mean_velocity
            max_lag = len(velocities) - 1
           # vacf = calculate_vacf(velocities_shifted, max_lag)
            vacf = calculate_vacf(velocities, max_lag)
            #vacf = calc_vacf(velocities)
            
            #print(vacf)
            # Normalize the VACF
            vacf = vacf / vacf[0]
            
            if np.isnan(vacf).any():
                continue
            print(vacf)
          
            all_track_vacf_out.append(vacf)
            track_ids.append(track_id)
            
    # Append the track lengths to the list
  
    
 
   
  

# plot histogram
# Plot VACF
print(len(all_track_vacf_out))
avg_vacf = np.mean(np.asarray(all_track_vacf_out), axis = 0)
std_vacf = np.std(all_track_vacf_out, axis = 0) #/ np.sqrt(len(all_track_vacf_out))
#std_vacf95 = np.percentile(all_track_vacf_out, 95, axis=0)


all_track_vacf_arr = np.asarray(all_track_vacf_out)    
vacf_norm_df = pd.DataFrame(all_track_vacf_arr.T)
time_lags = range(max_lag+1)
vacf_norm_df.insert(0,'Time', time_lags)
vacf_norm_df.to_csv(directory[:-5] + name +  '_mintracklen-{}'.format(min_frames) + '_vacf_norm.csv', index = False)
print(vacf_norm_df)

print(avg_vacf)

fig, axes = plt.subplots(figsize=(3,3), dpi=150) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
plt.axvline(x=1, linestyle='dashed', color='gray')
plt.axhline(y=0, linestyle='dashed', color='gray')
#for vacf in all_track_vacf_out:
    
    #plt.plot(range(max_lag+1), vacf, color='gray',alpha=0.01,)
plt.plot(range(max_lag+1), avg_vacf, marker='o', color='black', markersize=3, linewidth=1)
#plt.fill_between(range(max_lag+1), avg_vacf - std_vacf, avg_vacf + std_vacf, alpha=0.75
#                         ,color='gray')
plt.xlabel('Time Lag (s)')
plt.ylabel('Velocity Autocorrelation')


plt.xlim(0, 10)
plt.ylim(-.5,1.1)
fig.tight_layout()
plt.savefig(directory[:-5] + name + '_vacf.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_vacf.png', dpi=300) 
plt.show()


'''
VACF_t1 = np.asarray([item[1] for item in all_track_vacf_out])

avg_vacf1 = np.mean(VACF_t1, axis = 0)
std_vacf1 = np.std(VACF_t1, axis = 0) #/ np.sqrt(len(all_track_vacf_out))
percentile_vacf1 = np.percentile(VACF_t1,95, axis = 0)
print("tau 1: ", avg_vacf1, percentile_vacf1)

fig, axes = plt.subplots(figsize=(2.2, 1.8), dpi=150) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9

binwidth = 0.1
binBoundaries = np.arange(min(VACF_t1),max(VACF_t1), binwidth)
x, bins, p = plt.hist(VACF_t1, bins=binBoundaries, color='teal',
                      edgecolor='k',   weights = np.zeros_like(VACF_t1) + 1 / VACF_t1.size)#density=True)
                        
plt.xlabel('Cv (1 s)', fontsize=fontsize, ) 
plt.ylabel('Fraction of tracks', fontsize=fontsize)
fig.tight_layout()
plt.xlim(-.7, .7)
#plt.ylim(0,1.1)
fig.tight_layout()
plt.savefig(directory[:-5] + name + '_vacf_dist.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_vacf_dist.png', dpi=300) 
plt.show()
'''
