# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 13:59:39 2025

@author: azaldegc
"""

import seaborn as sns
import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import tifffile as tif
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import statistics as ss
import math as math
from skimage import (filters, measure)

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames



# calculated sing mol intensity
name = 'Hn_2'
label = 'rep_1'
pixsize = 0.049 # in microns


# load images
directory = sys.argv[1]
files = filepull(directory)
mask_files = [file for file in files if "_PhaseMask.tif" in file]
fluor_files = [file for file in files if 'fluor_normalized.tif' in file]
mask_files.sort()
fluor_files.sort()



# for each image
for image, mask_file in zip(fluor_files, mask_files):
  
    # read fluor and mask
    fluor = tif.imread(image) 
    mask = tif.imread(mask_file)

    # define cell regions from mask
    labels, n_labels = measure.label(mask, background=0, return_num=True)
    cells = measure.regionprops(labels, intensity_image = fluor)
    
    data_save = []

    # loop through each cell in the mask
    for ii, cell in enumerate(cells[:]):
    
        temp_mask = mask.copy()  # use temporary mask to prevent issues
        
        # define the cell label
        cell_origin = cell.centroid  # define the centroid of the cell
        cell_label = mask[int(cell_origin[0]), int(cell_origin[1])] # determine the label of cell
        print("cell label", cell_label)
        
        # determine the integrated cell intensity
        cell_summed_intensity = cell.image_intensity.sum()
        
        
        # determine the cell area
        cell_area = cell.area
        
        #print(cell_summed_intensity, cell_area, cell_summed_intensity / cell_area )
        data_save.append((image, name, label, cell_label, cell.major_axis_length*pixsize, 
                          cell.minor_axis_length*pixsize, cell.perimeter,
                          cell_area*(pixsize)**2, cell_area, 
                          cell_summed_intensity, 
                          cell_summed_intensity / cell_area))
        
    data_df = pd.DataFrame(data_save)
    data_df.columns = ['FILE', 'SAMPLE', 'LABEL', 'CELL_ID', 'CELL_LENGTH',
                   'CELL_WIDTH', 'CELL_PERIMETER','CELL_AREA', 'CELL_AREA_PIX',
                   'SUMMED_INTENSITY', 
                   'AVERAGE_INTENSITY']
    print(data_df)
    data_df.to_csv(image[:-4] + '_singlecell_intensities.csv')

        

    