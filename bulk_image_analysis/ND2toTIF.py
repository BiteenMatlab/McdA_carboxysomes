# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:11:30 2021

@author: azaldegc


Testing to see how to open nd2 files
"""

from nd2reader import ND2Reader
from nd2 import ND2File
import tifffile as tif
import sys
import numpy as np
import glob


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# foldername that contains all of the fits.mat files
directory = sys.argv[1]
n_channels = int(sys.argv[2])
timelapse = int(sys.argv[3])
# pull in all of the fits.nd2 files
nd2_files = filepull(directory)


for nd2 in nd2_files[:]:
    
    with ND2File(nd2) as images:
        #print(images.shape)
        stack = np.asarray(images)
        print(stack.shape)
        #images.iter_axes = 'zc'
        tiff_stack = [[] for i in range(int(n_channels))]
        channels = [i for i in range(n_channels)]
        #print(channels)
        #print(stack[:,0,:,:].shape)
        
        if n_channels == 1:
            
            #for ii,image in enumerate(stack[chan::n_channels]):
            if timelapse == 1:
                current_chan = stack[:,:,:]
                for mm in range(len(current_chan)):
                
                    fov = np.asarray(current_chan[mm])
                    tiff_stack[0].append(fov)
                     
            elif timelapse == 0:
                current_chan = stack
                fov = np.asarray(current_chan)
                tif.imsave(nd2[:-4] + '_fluor.tif', fov) 
                print(fov.shape)
            
            
        elif n_channels > 1: 
            for chan in channels:            
                #for ii,image in enumerate(stack[chan::n_channels]):
                    if timelapse == 1:
                        current_chan = stack[:,chan,:,:]
                        for mm in range(len(current_chan)):
                
                            fov = np.asarray(current_chan[mm])
                            tiff_stack[chan].append(fov)
                     
                    elif timelapse == 0:
                        current_chan = stack[chan] 
                        fov = np.asarray(current_chan)
                        tiff_stack[chan].append(fov)
                
                    
        #tiff = np.asarray(tiff_stack)
        #print(tiff.shape)

        #tif.imsave(nd2[:-4] + '_fluor.tif', tiff) 