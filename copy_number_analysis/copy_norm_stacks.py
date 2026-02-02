# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 17:11:57 2025

@author: azaldegc
"""


import sys
import numpy as np
import glob
import tifffile as tif
import matplotlib.pyplot as plt

# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# load directory with files
directory = sys.argv[1]
files = filepull(directory)
mov_files = [file for file in files if '_normalized.tif' not in file and 'fluor.tif' in file]



# load beam image
beam_file = sys.argv[2]

# read beam image
beam_img = tif.imread(beam_file) 

# subtract offset from beam image
beam_img_corrected = beam_img - 500 

# normalize the beam image
beam_img_norm = beam_img_corrected / np.max(beam_img_corrected)

plot_fig = False

# loop through each movie
for file in mov_files[:]:

    # read the movie
    mov_stack = tif.imread(file)
    print(mov_stack.shape)
    
    # normalize the movie
    mov_stack_norm = np.zeros(mov_stack.shape)
    
    if len(mov_stack.shape) > 2:
        for ii, frame in enumerate(mov_stack):
        
            mov_stack_norm[ii] = (frame - 500) / beam_img_norm
            
    elif len(mov_stack.shape) == 2:
        mov_stack_norm = (mov_stack - 500) / beam_img_norm
        
    if plot_fig == True:
        fig, ax = plt.subplots(ncols=3, figsize=(6, 3), 
                               sharey=True, sharex=True)
        ax[0].imshow(frame, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(beam_img_norm, cmap='gray')
        ax[1].set_title('Beam Image')
        ax[1].axis('off') 
        ax[2].imshow( mov_stack_norm[ii], cmap='gray')
        ax[2].set_title('Segmented cells')
        ax[2].axis('off')
        fig.tight_layout()
        plt.show()
             
        
        
     # save the movie
    tif.imwrite(file[:-4] + '_normalized.tif', mov_stack_norm)
        
    
print("Movies analyzed: ", len(mov_files))   



