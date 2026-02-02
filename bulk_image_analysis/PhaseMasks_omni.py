# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:03:39 2023

@author: azaldegc
"""

# import necessary packages
from skimage import (filters,  morphology, segmentation)
from cellpose_omni import models, plot
import matplotlib.pyplot as plt
import tifffile as tif
import sys
import numpy as np
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# from cellpose_omni import MODEL_NAMES
# from cellpose_omni import models, core, plot


def filepull(directory):
    '''Finds files in directory and returns filenames as a list'''

    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]

    return filenames


def segment_cells(model, filename, img, user_diam, PlotFig=False):
    """
    This function is used to segment bacterial cells in a phase contrast image
    using the Omnipose package and a pre-processing blurring step and post-
    processing removal of small segmented objects. 

    """

    blur_img = filters.gaussian(img, sigma=0.5)  # blur image before cellpose

    # channels = [0,0]
    chans = [0, 0]

    mask_threshold = 2
    verbose = 0  # turn on if you want to see more output
    # use_gpu = False  # defined above
    transparency = True  # transparency in flow output
    rescale = None  # give this a number if you need to upscale or downscale your images
    omni = True  # we can turn off Omnipose mask reconstruction, not advised
    # default is .4, but only needed if there are spurious masks to clean up; slows down output
    flow_threshold = 0
    resample = True  # whether or not to run dynamics on rescaled grid or original grid
    cluster = True  # use DBSCAN clustering

    masks, flows, styles = model.eval(blur_img, channels=chans, rescale=rescale, mask_threshold=mask_threshold,
                                      transparency=transparency, flow_threshold=flow_threshold, omni=omni,
                                      cluster=cluster, resample=resample, verbose=verbose)

   # io.masks_flows_to_seg(blur_img, masks, flows, filename, chans)

    # remove small objects and objects touching the imae border
    masks = morphology.remove_small_objects(masks, 100)
    masks = segmentation.clear_border(masks)
    
    # omnipose step-by-step plot
    if PlotFig == True:

        fig = plt.figure(figsize=(12, 5))
        plot.show_segmentation(fig, img, masks, flows[0], channels=chans)
        plt.tight_layout()
        plt.show()

    # returns segmentation and pre-processed image
    return masks, blur_img


# load files
directory = sys.argv[1] # foldername that contains all of the phase contrast images (.tif)
stack = False # if .tiff images are contain multi-channel or time-lapse data

files = filepull(directory) # pull files from directory

# tif_files = [file for file in files if '_phase_' in file and "PhaseMask.tif" not in file] 
tif_files = [file for file in files if "PhaseMask.tif" not in file] # only use .tif files that are not segmentation output

print("{} .tif files found".format(len(tif_files))) # print number of files 
tif_files.sort() # sort files

plot_fig = True # to plot cell segmentation output


# select model for segmentation, use default omnipose phase
model_name = 'bact_phase_omni'
model = models.CellposeModel(gpu=False, model_type=model_name)

# loop through each file

for ii, file in enumerate(tif_files):

    print(file) # print file name

    if stack == True: # if image is a stack

        print(tif.imread(file).shape) # print shape
        img = tif.imread(file)[2][0] # edit here what channel or time point is needed for segmentation
        print(np.shape(img)) # to check that selected frame is of correct dimensions
        tif.imwrite(file[:-4] + '_phase.tif', img) # generate phase contrast image for omnipose GUI editing

    elif stack == False: # if file is a sinlge image

        img = tif.imread(file) # read file

    # invert image twice
    img_inv = np.invert(img)
    img_inv = np.invert(img_inv)
    
    # segment cells
    segmented_cells, img_processed = segment_cells(model, file, img_inv,
                                                 user_diam=None)

    # save segmentation image
    tif.imwrite(file[:-4] + '_PhaseMask.tif',
                segmented_cells)

    # display results
    if plot_fig == True:
        fig, ax = plt.subplots(ncols=3, figsize=(6, 3),
                               sharey=True, sharex=True)
        ax[0].imshow(img, cmap='gray')
        ax[0].contour(segmented_cells, z=0, levels=1,
                      linewidth=0.05, colors='r')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        ax[1].imshow(img_processed, cmap='gray')
        ax[1].contour(segmented_cells, z=0, levels=1,
                      linewidth=0.05, colors='r')
        ax[1].set_title('Pre-processed')
        ax[1].axis('off')
        ax[2].imshow(segmented_cells, cmap='turbo')
        ax[2].set_title('Segmented cells')
        ax[2].axis('off')

        fig.tight_layout()
        plt.show()
        
    print(file[:-4] + '_PhaseMask.tif') # indicate which file was created