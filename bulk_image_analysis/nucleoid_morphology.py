# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:22:13 2024

@author: azaldegc
"""

# import packages
import tifffile as tif
import numpy as np
import pandas as pd
from skimage import (measure, segmentation, morphology, filters,)
import matplotlib.pyplot as plt
from lmfit import Model
import glob
import sys


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


def get_pixel_intensities(image, coords):
    """
    To get pixel intensities values associated with an image region    

    Parameters
    ----------
    image : 2D array ; image of interest.
       
    coords : list ; list of coordinates/tuples (x, y).

    Returns
    -------
    pix_intensities : list ; list of pixel intensities
    
    """

    pix_intensities = []
    for coord in coords:
        pix_intensities.append(image[coord[0], coord[1]])

    return pix_intensities


def get_cell_inner_rim(region_mask, original_image):
    """
    To get the inner rim pixel intensities of an object region. 
    Used to determine the cell background fluorescence signal
    
    Parameters
    ----------
    region_mask : 2D array ; mask of region of interest.
    original_image : 2D array ; image of interest.
       
    Returns
    -------
    inner_border_pixels : array ; pixel intensities associated with the inner rim of the cell

    """

    # Define the inner cell rim boundary mask
    boundary_mask = segmentation.find_boundaries(
        region_mask, connectivity=1, mode='inner')

    # Extract pixel values from the inner border
    inner_border_pixels = original_image[boundary_mask]

    return inner_border_pixels


def get_cell_background(pixel_ints, sub='one', plot=False):
    """
    To determine the background signal within a cell region. 
    Distribution of pixel intensities of the inner rim are fit to a Gaussian
    model.

    Parameters
    ----------
    pixel_ints : list ; pixel intensities
    sub : number of Gaussians to fit to ; The default is 'one'. Can only do up
    to 2. 
    plot : bool, optional ; To plot the fit on the distribution.
    Returns
    -------
    float ; mean of Gaussian fit + 2 standard deviations

    """

    params_1 = (.005, 500, 50)
    params_2 = (.005, 25, 50, .005, 100, 50)

    hist, bin_edges = np.histogram(pixel_ints, bins=10, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))

    # Define the double Gaussian model
    def double_gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
        return (amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2)) +
                amp2 * np.exp(-(x - cen2)**2 / (2 * sigma2**2)))

    model = Model(gaussian)

    if sub == 'one':
        params = model.make_params(
            amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])
    elif sub == 'two':
        params = model.make_params(amp1=params_2[0], cen1=params_2[1], sigma1=params_2[2],
                                   amp2=params_2[3], cen2=params_2[4], sigma2=params_2[5])

    # Perform the fit
    result = model.fit(hist, x=bin_centers, params=params)

    # get peak of first gaussian

    if plot == True:
        # Plot the histogram and the fitted Gaussian
        plt.hist(pixel_ints, bins=10, density=True,
                 alpha=0.7, label='Histogram')
        plt.plot(bin_centers, result.best_fit, 'r-', label='Gaussian Fit')
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.show()

    return result.params['cen1'].value + 2*result.params['sigma1'].value

# script


pix_size = 0.066  # in microns
plotfig = False  # plot background subtraction for each cell

name = 'Hn_2'

# load in files
directory = sys.argv[1]
files = filepull(directory)

mask_files = [file for file in files if "PhaseMask.tif" in file]
fluor_files = [file for file in files if "PhaseMask.tif" not in file]

mask_files.sort()
fluor_files.sort()

data = []

for mm in range(len(mask_files)):

    # assign correct channel or time point in stack
    DAPI_file = fluor_files[mm]

    mask = segmentation.clear_border(tif.imread(mask_files[mm]))  # read mask

    # read DAPI image, first channel
    dapi_img = tif.imread(DAPI_file)[0][0]  # read the correct frame
    print(dapi_img.shape)  # to check that image is correct dimensions

    # identify cells based on cell mask
    labels, n_labels = measure.label(mask, background=0, return_num=True)
    cells = measure.regionprops(labels)

    ex = 0  # how many pixels to expand the cell bounding box by in each direction

    # run through each cell in the mask
    print("Number of cells to analyze: ", len(cells))
    for kk, cell in enumerate(cells[:]):

        temp_mask = mask.copy()  # to avoid issues

        cell_label = int(mask[int(cell.centroid[0]), int(cell.centroid[1])])

        # cell IDs to ignore if needed
        ignore_cells = []
        if cell_label in ignore_cells:
            continue

        # get mask region, but exclude surrounding cells in the roi
        cell_mask = temp_mask[cell.bbox[0]-ex:cell.bbox[2] +
                              ex, cell.bbox[1]-ex:cell.bbox[3]+ex]
        # remove any other cells in the ROI
        cell_mask[cell_mask != cell_label] = 0
        cell_mask[cell_mask == cell_label] = 1  # change cell value to 1
        img_bg = dapi_img

        # crop fluorescent channel for visualization purposes
        cell_dapi = img_bg[cell.bbox[0]-ex:cell.bbox[2] +
                           ex, cell.bbox[1]-ex:cell.bbox[3]+ex]
        dapi_by_mask = cell_mask * cell_dapi

        # gather all of the pixel intensities within the cell or inner cell border
        # intensities = get_pixel_intensities(img_bg, cell.coords)
        intensities = get_cell_inner_rim(cell_mask, cell_dapi)

        # determine background intensity inside of the cell
        cell_bg_int = get_cell_background(intensities, plot=False)

        # background subtract
        cell_dapi_bg_correct = cell_dapi - cell_bg_int
        intensities_bg_corr = intensities - cell_bg_int

        # threshold
        # thresh = filters.threshold_otsu(dapi_by_mask_bg_corr)
        cell_dapi_bg_correct[cell_dapi_bg_correct < 0] = 0
        corrected_by_mask = cell_mask*filters.gaussian(cell_dapi_bg_correct, 1)

        # identify nucleoid by yen thresholding
        thresh = filters.threshold_yen(corrected_by_mask)
        binary = corrected_by_mask > thresh
        binary = morphology.remove_small_objects(
            binary, 50)  # remove small objects

        cell_area = cell.area * pix_size**2  # calculate cell area in microns-squared
        cell_length = cell.axis_major_length * \
            pix_size  # calculate cell length in microns

        label_seg_nucleoid = measure.label(binary)
        nucleoids = measure.regionprops(label_seg_nucleoid)

        # in case that there are multiple nucleoids identified
        # it is possible in pre-divisional cells
        for ii, nucleoid in enumerate(nucleoids):

            nucleoid_length = nucleoid.axis_major_length * pix_size
            nucleoid_area = nucleoid.area * pix_size**2

            if len(nucleoids) > 1:
                multi_nucleoid_bool = True
            elif len(nucleoids) == 1:
                multi_nucleoid_bool = False

            # remove erroneous nucleoid detection
            # only use nucleoid that are not the same size as the cell
            if nucleoid_length / cell_length <= 1:

                data.append([DAPI_file, name, cell_label, cell_length, cell_area,
                             nucleoid_length, nucleoid_area, multi_nucleoid_bool])

        # plot cell by cell
        if plotfig == True:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(7, 3))
            ax = axes.ravel()
            ax[0].set_title('cell #: {}'.format(cell_label))
            ax[0].imshow(cell_dapi, cmap='inferno')
            ax[0].contour(cell_mask, levels=0, colors='w')

            ax[1].set_title('cell #: {}'.format(cell_label))
            ax[1].imshow(corrected_by_mask, cmap='inferno', vmin=0)
            ax[1].contour(binary, levels=0, colors='cyan')
            ax[1].contour(cell_mask, levels=0, colors='w')

            fig.tight_layout()
            plt.show()

# save the cell ID, total foci per cell, and locations in a csv
data_df = pd.DataFrame(data)
data_df.columns = ['File', 'Sample', 'Cell_ID', 'Cell_Length', 'Cell_Area',
                   'Nucleoid_Length', 'Nucleoid_Area', 'Multi_Nucleoid']
print(data_df)
data_df.to_csv(directory[:-5] + name + '_nucleoid_analysis.csv')

print("Done analyzing cells", len(data))