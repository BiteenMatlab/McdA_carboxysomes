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


# Define data name or label
name = 'Hn_441_80_TL220ms'
# Define the integration time for imaging (in seconds)
t_int = 0.08
# Define the time delay between frames in seconds
t_delay = 0.14
# Define minimum track length (in frames)
min_frames = 4
# Define maximum frame allowed within a track
max_gap = 1
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


def has_gap(values):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > (max_gap+1):
            return True
    return False

directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'tracks' in file]
# Initialize a list to store the track lengths from all files

all_track_radius_of_confinement = []
track_ids = []

# load through each file
for file in track_files[36:]:
    
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
        if n >= min_frames and gap == False:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['LOC_C'].to_numpy()[:]
            track_y_coords = track_data['LOC_R'].to_numpy()[:]
            #print(len(track_x_coords), len(track_y_coords))
                
           
            all_track_radius_of_confinement.append(calculate_confinement_radius(track_x_coords, track_y_coords))# / n*(t_int+t_delay))
            track_ids.append(track_id)
    # Append the track lengths to the list
    #all_track_radius_of_confinement.extend(radius_of_confinements)


all_track_radius_of_confinement = np.asarray(all_track_radius_of_confinement)
print(len(all_track_radius_of_confinement), np.mean(all_track_radius_of_confinement) )
bootstraps = 100
medians = []
for n in range(bootstraps):
    bootstrapped_data =  np.random.choice(all_track_radius_of_confinement, replace=True, 
                             size=int(len(all_track_radius_of_confinement)*1))
    medians.append(np.median(bootstrapped_data))
print("mean avg +/- stdev", np.mean(medians), np.std(medians))

mobile_tracks = all_track_radius_of_confinement[all_track_radius_of_confinement > 0.031]
print("number of mobile tracks", len(mobile_tracks), len(mobile_tracks) / len(all_track_radius_of_confinement))

# plot histogram
binwidth = 0.005
binBoundaries = np.arange(min(all_track_radius_of_confinement),
                          max(all_track_radius_of_confinement), binwidth)

fig, axes = plt.subplots( figsize=(5,5), 
                                 dpi=100)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
x, bins, p = plt.hist(all_track_radius_of_confinement,  color='gray',
                     edgecolor='k', bins=binBoundaries,
                     weights = np.zeros_like(all_track_radius_of_confinement) + 1 / all_track_radius_of_confinement.size)
plt.axvline(x=.031, linestyle='dashed', color='gray')
plt.xlim(0,.5)
plt.ylim(0, 0.7)
plt.xlabel('Radius of Confinement')
plt.ylabel('Normalized Frequency')
plt.title('Confinement Radius Histogram')
fig.tight_layout()
#plt.savefig(directory[:-5] + name + '_confinement.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_confinement.png', dpi=300)
plt.show()




labels = [name for x in range(len(all_track_radius_of_confinement))]
rad_confinement_df = pd.DataFrame({'name': labels,
                                   'track_id': track_ids,
                                   'Radius of Confinement': all_track_radius_of_confinement})
#rad_confinement_df.to_csv(directory[:-5] + name + '_radius_confinements.csv',index = False)


