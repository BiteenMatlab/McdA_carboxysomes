# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 10:37:33 2025

@author: azaldegc
"""

import sys
import numpy as np
import glob
import tifffile as tif
import matplotlib.pyplot as plt
from skimage import (feature)
import pandas as pd


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def detect_spots(image, plot=True):
    
    # pre process to remove super bright pixels
    
    img_median = np.median(image)
    image[image > 65535] = img_median
    
    # detect spots
    spots_log = feature.blob_log(image, min_sigma = 2.5, max_sigma = 3.5, 
                                 threshold = 2000, overlap = 0.5)
    
    
    # plot spot detection to verify parameters work
    if plot == True:# and len(spots_log) > 0:
        fig, axes = plt.subplots(1, 1, figsize=(3, 3), sharex=False, sharey=False)
        
        axes.imshow(image, cmap='gray', vmin=500, vmax=20000)
        axes.axis("off")
        for nn,blob in enumerate(spots_log):
            y, x, r = blob
            c1 = plt.Circle((x, y), r, linewidth=1, fill=False, color='cyan')
            axes.add_patch(c1)
            
        plt.tight_layout()
        plt.show()
        
        
    # return list of spots
        
    return spots_log

# function to make a mask for each spot
def create_circle_mask(h,w, center, radius):
    
    Y, X =np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    
    return mask
    
# load directory with files
directory = sys.argv[1]
files = filepull(directory)
mov_files = [file for file in files if '_normalized.tif' in file]

# loop through each movie
for file in mov_files[:]:
    print(file)

    # read the movie
    mov_stack = tif.imread(file)
    
    # analyze the first image to detect spots
    spot_coords = detect_spots(mov_stack[0])
    
    traces = []
    
    for spot in spot_coords:
        
        y, x, r = spot
        blob_mask = np.zeros((mov_stack[0].shape[0], mov_stack[0].shape[1]))
        blob_grid = create_circle_mask(mov_stack[0].shape[0], mov_stack[0].shape[1],
                                       center = (x,y), radius=r)
        blob_mask[blob_grid] = 1
        
        spot_trace = []
        
        for frame in mov_stack:
            masked_fluor = blob_mask * frame
            out_arr = masked_fluor[np.nonzero(masked_fluor)]
            spot_trace.append(sum(out_arr))
            print(len(out_arr))
     
        traces.append(spot_trace)
        
        plot_trace = False
        if plot_trace == True:
            
            plt.plot(range(len(spot_trace)), spot_trace)
            changepoints = []
            plt.show()
                
    print("Collected traces: ", len(traces))
    
    # save traces for that file
    traces_arr = np.asarray(traces).T
    data_df = pd.DataFrame(traces_arr)
    #print(data_df)
    data_df.to_csv(file[:-4] + '_spot_traces.csv')
            