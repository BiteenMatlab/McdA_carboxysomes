# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:57:14 2024

@author: azaldegc
"""


# import modules
import sys
import numpy as np
import glob
from lmfit import Model
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import math

# Define data name or label
name = 'fixed_80ms'
# Define the integration time for imaging (in seconds)
t_int = 0.08
# Define the time delay between frames in seconds
t_delay = 0
# Define minimum track length (in frames)
min_frames = 4
# Define the pixel size in nanometers
pixel_size = 49
# Define maximum frame allowed within a track
max_gap = 1

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def calculate_confinement_radius(x_coords, y_coords):
    # Calculate centroid
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    
    # Calculate distances to centroid
    distances = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
    
    # Calculate confinement radius
    confinement_radius = np.mean(distances)*pixel_size
    
    return confinement_radius


def has_gap(values):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > max_gap:
            return True
    return False

def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))
    
    
directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'tracks' in file]
# Initialize a list to store the track lengths from all files

all_track_radius_of_confinement = []
track_ids = []

# load through each file
for file in track_files:
    
    # read csv as dataframe
    df = pd.read_csv(file, header=0)
    # Calculate the radius of confinement
    radius_of_confinements = []
    # loop through each track
    for track_id in df["TRACK_N"].unique():
        # load track data
        track_data = df[df["TRACK_N"] == track_id]
        n = len(track_data)
        
        gap = gap = has_gap(track_data["FRAME_N"].tolist())
        
        # calculate radius of confinement if track is min_frames long
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['LOC_C'].to_numpy()[:]
            track_y_coords = track_data['LOC_R'].to_numpy()[:]
            #print(len(track_x_coords), len(track_y_coords))
                
            radius = calculate_confinement_radius(track_x_coords, track_y_coords)
            if radius > 0:
                radius_of_confinements.append(radius)
                track_ids.append(track_id)
    # Append the track lengths to the list
    all_track_radius_of_confinement.extend(radius_of_confinements)


all_track_radius_of_confinement = np.asarray(all_track_radius_of_confinement)
#print(all_track_radius_of_confinement.shape)
print(len(all_track_radius_of_confinement))
binwidth = 2.5
binBoundaries = np.arange(min(all_track_radius_of_confinement),
                          max(all_track_radius_of_confinement), binwidth)

params_1 = (1,25, 5)
hist, bin_edges = np.histogram(all_track_radius_of_confinement, bins=binBoundaries, 
                               weights = np.zeros_like(all_track_radius_of_confinement) +
                               1 / all_track_radius_of_confinement.size)# density=True)  
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
model = Model(gaussian)
params = model.make_params(amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])

result = model.fit(hist, x=bin_centers, params=params)

fig, axes = plt.subplots(figsize=(3.1,3), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
x, bins, p = plt.hist(all_track_radius_of_confinement,  color='gray',
                     edgecolor='k', bins=binBoundaries,
                     weights = np.zeros_like(all_track_radius_of_confinement) + 1 / all_track_radius_of_confinement.size)

print(result.params['cen1'].value, result.params['sigma1'].value)
x_smooth = np.linspace(min(all_track_radius_of_confinement),
                         max(all_track_radius_of_confinement), 1000)
yfit = gaussian(x_smooth, result.params['amp1'].value ,result.params['cen1'].value, result.params['sigma1'].value)
plt.plot(x_smooth, yfit, 'k--', label='Gaussian Fit')
plt.xlabel('Radius of Confinement')
plt.ylabel('Normalized Frequency')
plt.xlim(0,50)
plt.ylim(0, 0.2)
plt.xticks([0, 25, 50])
#plt.title('Confinement Radius Histogram')
fig.tight_layout()
plt.savefig(directory[:-5] + name + '_uncertainty.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_uncertainty.png', dpi=300)
plt.show()




labels = [name for x in range(len(all_track_radius_of_confinement))]
rad_confinement_df = pd.DataFrame({'name': labels,
                                   'track_id': track_ids,
                                   'Radius of Confinement': all_track_radius_of_confinement})
rad_confinement_df.to_csv(directory[:-5] + name + '_radius_confinements.csv',index = False)


