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

import pandas as pd
import matplotlib.pyplot as plt


# Define data name or label
name = 'Hn_14'
# Define the integration time for imaging (in seconds)
t_int = .08
# Define the time delay between frames in seconds
t_delay = 0.14
# Define minimum track length (in frames)
min_frames = 40
# Define the pixel size in microns
pixel_size = 0.049

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

directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'spots' in file]
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
    for track_id in df["TRACK_ID"].unique():
        # load track data
        track_data = df[df["TRACK_ID"] == track_id]
        n = len(track_data)
        
        # calculate radius of confinement if track is min_frames long
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['POSITION_X'].to_numpy()[:]
            track_y_coords = track_data['POSITION_Y'].to_numpy()[:]
            #print(len(track_x_coords), len(track_y_coords))
                
           
            radius_of_confinements.append(calculate_confinement_radius(track_x_coords, track_y_coords))
            track_ids.append(track_id)
    # Append the track lengths to the list
    all_track_radius_of_confinement.extend(radius_of_confinements)

# plot histogram
histogram, bins = np.histogram(all_track_radius_of_confinement, 
                               bins=np.arange(0, max(all_track_radius_of_confinement), 0.005))
normalized_histogram = histogram / np.sum(histogram)
plt.bar(bins[:-1], normalized_histogram, width=0.005, alpha=0.7)
plt.xlabel('Radius of Confinement')
plt.ylabel('Normalized Frequency')
plt.title('Confinement Radius Histogram')
plt.show()




labels = [name for x in range(len(all_track_radius_of_confinement))]
rad_confinement_df = pd.DataFrame({'name': labels,
                                   'track_id': track_ids,
                                   'Radius of Confinement': all_track_radius_of_confinement})
rad_confinement_df.to_csv(directory[:-5] + name + '_radius_confinements.csv',index = False)


