# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 17:13:42 2024

@author: azaldegc

Script for batch run of detection and localization of carboxysomes in H. neapolitanus

"""

# import packages
import tifffile as tif
import numpy as np
import pandas as pd
from skimage import (measure, feature, segmentation, filters)
import matplotlib.pyplot as plt
from lmfit import Model
from scipy.optimize import curve_fit
import math
import glob
import sys


def create_circle_mask(h, w, center, radius):
    """
    To create a mask for the detected carboxysome focus

    Parameters
    ----------
    h : int ; Image region height or number of rows.
    w : int ; Image region width or number of columns.
    center : tuple (x, y); center coordinate of focus 
    radius : int; radius of focus
       
    Returns
    -------
    mask : 2D array; mask for focus

    """

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X-center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

def distance(p1, p2):
    """
    To calculate the distance between two coordinates.

    Parameters
    ----------
    p1 : tuple (x, y) ; start point.
    p2 : tuple (x, y) ; end point.

    Returns
    -------
    float ; distance between p1 and p2
    """
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def nearest_neighbor(points):
    """
    To sort coordinates by the shortest possible path. Each point is sorted 
    such that they are next to its nearest point without repetition.
    This is how the nearest neighbor distances are identified.

    Parameters
    ----------
    points : list ; list of coodinates
    
    Returns
    -------
    path : list ; list of sorted coordinates
    """
    
    n = len(points)
    visited = [False] * n
    path = []
    current_point = points[0]  # Start from the first point

    for _ in range(n - 1):
        path.append(current_point)
        visited[points.index(current_point)] = True
        nearest_dist = float('inf')
        nearest_point = None

        for point in points:
            if not visited[points.index(point)]:
                dist = distance(current_point, point)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_point = point

        current_point = nearest_point

    path.append(current_point)
    return path

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
    boundary_mask = segmentation.find_boundaries(region_mask, connectivity=1, mode='inner')

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


def moments(data, num_peaks, x_coord, y_coord):
    '''
        Determines guess parameters for each particle fit
        
        Parameters
        ----------
        data: array, processed image
        num_peaks: int, number of particles to fit to 
        x_coord: array, corresponding x-coordinate location for each guess
        y_coord: array, corresponding y-coordinate location for each guess
        
        Returns
        ----------
        packs: array, wrap of guess parameters for all particles
    '''

    global total
    total = data.sum()  # add all pix values in img
    Y, X = np.indices(data.shape)  # array of pix coords

    # stdev defined by diffration limited size of particle
    # size is the width it should be at 20% of the peak
    sigma = dfrlmz / (2*np.sqrt(2*np.log(5)))

    # Amplitude and Offset
    amplitude = data.max()

    # Generate guess parameters for target number of guesses
    packs = []

    # for each particle, wrap guess parameters
    for i in range(num_peaks):
        pack = [x_coord[i], y_coord[i], sigma, amplitude]
        packs += pack
    packs = np.array(packs)

    # If there is a NaN value in guess parameters convert it to 1
    packs[np.isnan(packs)] = 1

    return packs


def getParamBounds(data, num_peaks):
    '''
        Determines guess parameter bounds for each particle fit
        
        Parameters
        ----------
        data: array, processed image
        num_peaks: int, number of particles to fit to 
        x_coord: array, corresponding x-coordinate location for each guess
        y_coord: array, corresponding y-coordinate location for each guess
        
        Returns
        ----------
        packs: array, wrap of guess parameters for all particles
    '''
    # max location is bounded by image, min location is bounded by image
    y0_max, x0_max = data.shape
    y0_min, x0_min = 0, 0

    # sigma min and max is target width / or * by user param, stdtol
    sigma_min = (dfrlmz / (2*np.sqrt(2*np.log(5)))) / stdtol
    sigma_max = (dfrlmz / (2*np.sqrt(2*np.log(5)))) * stdtol

    # amplitude min is 10% of max peak
    Amp_min = data.max() * 0.01
    Amp_max = 100000

    # group min and max bounds
    boundmin = x0_min, y0_min, sigma_min, Amp_min
    boundmax = x0_max, y0_max, sigma_max, Amp_max

    BoundMins = ()
    BoundMaxs = ()
    # wrap min and max bounds
    for i in range(int(num_peaks)):
        BoundMins += boundmin
        BoundMaxs += boundmax
    boundPar = (BoundMins, BoundMaxs)

    return boundPar


def gaussian_2d(x, y, x0, y0, sigma, A):
    """
    2D Gaussian model
    """   
    return A*np.exp(-((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))


def _gauss(M, *args):
    """
    Parameters
    ----------
    M : (2, N) array ; M is a (2,N) array where N is the total number of data 
    points in Z, which will be ravelled to 1D.
    *args : various ; 

    Returns
    -------
    arr : array ; parameters
    """
    

    # Callable that is passed to curve_fit. 
    x, y = M
    arr = np.zeros(x.shape)
    for i in range(len(args)//4):
        arr += gaussian_2d(x, y, *args[i*4:i*4+4])

    return arr


# function to look at multiple files
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


# Script 

pix_size = 0.066  # in microns
plot = False  # plot the carboxysome detection step
bg_subtract = False # to perform background subtraction

label = 'Hn_404' # sample label

# detection parameters to play with
global dfrlmz, stdtol

# parameters for initial spot detection
detect_min_sigma = 1.05  # lower radius limit for focus
detect_max_sigma = 2.  # higher radius limit for focus
denoise_radius = 3
denoise_amount = 50
gauss_blur = 0.5
detect_threshold = .2  # how bright a focus needs to be to detect
overlap = 0.5  # lower value the more less overlap allowed
neighbor_size = 1*pix_size

# parameters for Gaussian fitting of detected spots
dfrlmz = 3 # diameter of spot in pixels
stdtol = 3 # tolerance of standard deviation of the Gaussian fit


# load in files
directory = sys.argv[1]
files = filepull(directory)

mask_files = [file for file in files if "PhaseMask.tif" in file]
fluor_files = [file for file in files if "mTQ_aligned.tif" in file]

mask_files.sort()
fluor_files.sort()

for mm in range(len(mask_files)):

    cb_file = fluor_files[mm] # carboxysome image stack

    # read mask
    mask_main = segmentation.clear_border(tif.imread(mask_files[mm]))

    cb_mov = tif.imread(cb_file) # read CB stack

    # define cell regions from mask
    labels, n_labels = measure.label(mask_main, background=0, return_num=True)
    cells = measure.regionprops(labels)
    
    # initiate lists to store data
    cbs_props = []
    cbs_quant = []

    ex = 1  # how many pixels to expand the cell bounding box by in each direction

    # run through each cell in the mask
    print("Number of cells to analyze: ", len(cells))
    for ii, cell in enumerate(cells[:]):

        temp_mask = mask_main.copy()  # use temporary mask to prevent issues

        cell_origin = cell.centroid  # define the centroid of the cell
        cell_label = mask_main[int(cell_origin[0]), int(
            cell_origin[1])]  # determine the label of cell
        print("cell label", cell_label)

        # crop the mask to the specific cell
        cell_mask = temp_mask[cell.bbox[0]-ex:cell.bbox[2] +
                              ex, cell.bbox[1]-ex:cell.bbox[3]+ex]
        # remove any other cells in the ROI
        cell_mask[cell_mask != cell_label] = 0
        cell_mask[cell_mask == cell_label] = 1  # change cell value to 1

        # if time-lapse image stack
        if len(np.shape(cb_mov)) == 3:
            
            particle_count = []
            particle_coords = []
            
            # iterate through each frame
            for nn, frame in enumerate(range(len(cb_mov))):
                
                cb_img = cb_mov[nn] # define current frame
                # crop the cb image as before
                cell_fluor = cb_img[cell.bbox[0]-ex:cell.bbox[2] +
                                    ex, cell.bbox[1]-ex:cell.bbox[3]+ex]

                ###############################################
                # pre processing to best localize carboxysomes
                # feel free to play with the numbers in these functions to best detect carboxysomes

                # first - backgroud subtraction, smaller radius work best but 1 is too small
                # optional, for carboxysomes I did not use it
                if bg_subtract == True:
                    intensities = get_cell_inner_rim(cell_mask, cell_fluor)
                    cell_bg_int = get_cell_background(intensities, plot=False)
                    cell_bg_correct = cell_fluor - cell_bg_int

                elif bg_subtract == False:
               
                    cell_bg_correct = cell_fluor
                    
                # image sharpening
                cell_bg_correct = filters.unsharp_mask(cell_fluor, 
                                                       radius=denoise_radius,
                                                       amount=denoise_amount)
                
                # blur sharpened image
                cell_bg_correct = filters.gaussian(cell_bg_correct, gauss_blur)
                
                # merge processed image with cell mask
                cb_by_mask = cell_bg_correct * cell_mask
                cb_by_mask[cb_by_mask <= 0] = 0

                global y_size, x_size
                yy, xx = np.mgrid[0:cb_by_mask.shape[0], 0:cb_by_mask.shape[1]]
                y_size, x_size = cb_by_mask.shape

                # detect carboxysomes using Laplacian of Gaussian function
                blobs_log = feature.blob_log(cb_by_mask, min_sigma=detect_min_sigma,
                                             max_sigma=detect_max_sigma, 
                                             threshold=detect_threshold, 
                                             overlap=overlap)

                # count the number of detected spots
                particle_count.append(len(blobs_log))
                particle_coords.append(blobs_log)

                # if not spots detected skip
                if len(blobs_log) == 0:
                    continue

        # determine frame with highest particle count
        frame_to_use = particle_count.index(max(particle_count))
        blobs_log = particle_coords[frame_to_use]
        print("Particle guess : ", len(blobs_log), frame_to_use)
        cb_img = cb_mov[frame_to_use]
        
        # perform the pre-processing steps again
        cell_fluor = cb_img[cell.bbox[0]-ex:cell.bbox[2] +
                            ex, cell.bbox[1]-ex:cell.bbox[3]+ex]
        cell_bg_correct = filters.unsharp_mask(cell_fluor, 
                                               radius=denoise_radius, 
                                               amount=denoise_amount)
        cell_bg_correct = filters.gaussian(cell_bg_correct, gauss_blur)
        cb_by_mask = cell_bg_correct * cell_mask
        cb_by_mask[cb_by_mask <= 0] = 0

        # fit detected spots for sub pixel localization
        tot_peaks, x_loc, y_loc = len(
            blobs_log), blobs_log[:, 1], blobs_log[:, 0]
        # determine guess paramaters using guess peaks locations
        params_guess = moments(cb_by_mask, tot_peaks, x_loc, y_loc)
        # determine paramater bounds
        param_bounds = getParamBounds(cb_by_mask, tot_peaks)

        # flatten x,y data
        xdata = np.vstack((xx.ravel(), yy.ravel()))
        data = cb_by_mask
        # fit to gaussian
        try:
            popt, pcov = curve_fit(_gauss, xdata, data.ravel(),
                                   params_guess, bounds=param_bounds,
                                   maxfev=10000)
        except:
            continue

        fit = np.zeros(data.shape)
        for i in range(len(popt)//4):
            fit += gaussian_2d(xx, yy, *popt[i*4:i*4+4])


        # sort carboxysomes by coordinates so thast they are adjacent to one another
        particles = np.reshape(popt, (-1, 4))
       
        sorted_particles = sorted(particles, key=lambda x: x[0])

        sorted_coordinates = [(blob[0], blob[1], blob[2], blob[3])
                              for blob in particles]
        
        # filter particle detection by a minimum radius
        # remove particles with radius less than 1 pixel
        pop_index = []
        for num, blob in enumerate(sorted_coordinates):
            if blob[2] < 1:
                pop_index.append(num)
       
        pop_index.reverse()
        if len(pop_index) > 0:
            for index in pop_index:
                sorted_coordinates.pop(index)
        
        # sort coordinates by nearest neighbor if more than one particle
        if len(particles) > 1:
            ordered_coordinates = nearest_neighbor(sorted_coordinates)

        else:
            ordered_coordinates = sorted_coordinates

        if len(particles) > 0:
            # save the cell ID, # of carboxysomes, and their locations
            cbs_props.append((cb_file, label, cell_label,
                              cell.major_axis_length*pix_size, cell.area*pix_size,
                              len(particles), len(particles) /
                              (cell.major_axis_length*pix_size),
                              ordered_coordinates))

        # for each carboxysome detected, measure it's properties
        if len(particles) > 0:
            for jj, blob in enumerate(ordered_coordinates):

                blob_mask = np.zeros(
                    (cell_fluor.shape[0], cell_fluor.shape[1]))

                x, y, r, a = blob  # carboxysome coordinate and radius

                # calculate the pairwise distances for each carboxysomes (in pixels)
                # it will have repeated values for redundant measurements
                if jj < len(ordered_coordinates) - 1:
                    pairwise_distance = distance(
                        blob, ordered_coordinates[jj+1])*pix_size
                else:
                    pairwise_distance = None

                # mask each carboxysome to determine its area and total intensity
                blob_grid = create_circle_mask(
                    cell_fluor.shape[0], cell_fluor.shape[1], center=(x, y), radius=1.5*r)
                blob_mask[blob_grid] = 1
                masked_fluor = blob_mask * cell_fluor
                out_arr = masked_fluor[np.nonzero(masked_fluor)]

                # save cell label, total cb in cell, cb label,cb radius, cb area, cb total intensity, pairwise distances
                cbs_quant.append((cb_file, label, cell_label, len(blobs_log), jj+1, r, len(
                    out_arr), sum(out_arr), pairwise_distance, cell.major_axis_length*pix_size))

        # here we plot each step of the preprocessing before detection
        # red circles are the detected carboxysomes
        # if false positives - adjust detection parameters or take note of cells to exclude them from further analysis/plotting
        if plot == True and len(blobs_log) > 0:
            fig, axes = plt.subplots(1, 4, figsize=(
                9, 3), sharex=False, sharey=False)
            ax = axes.ravel()

            ax[0].imshow(cell_fluor, cmap='inferno')
            ax[1].imshow(cell_bg_correct, cmap='inferno', vmin=0)
            ax[2].imshow(cell_bg_correct, cmap='inferno', vmin=0)
            ax[3].imshow(cb_by_mask, cmap='inferno', vmin=0)
            ax[0].contour(cell_mask, levels=0, colors='w')
            ax[1].contour(cell_mask, levels=0, colors='w')
            ax[2].contour(cell_mask, levels=0, colors='w')
            ax[3].contour(cell_mask, levels=0, colors='w')

            for nn, blob in enumerate(sorted_coordinates):
                x, y, r, a = blob
                c1 = plt.Circle((x, y), 1.5*r, linewidth=1,
                                fill=False, color='cyan')
                c2 = plt.Circle((x, y), 1.5*r, linewidth=1,
                                fill=False, color='cyan')
                ax[0].add_patch(c1)
                ax[3].add_patch(c2)
            plot_titles = ["original " + str(cell_label), 
                           "background corrected", "denoised", 
                           "blurred", "all"]
            for hh, axx in enumerate(ax):
                axx.set_title(plot_titles[hh])
               
            plt.tight_layout()
            plt.show()

    # save the cell ID, total cbs per cell, and locations in a csv
    cells_df = pd.DataFrame(cbs_props)
    cells_df.columns = ['FILE', 'SAMPLE', 'CELL_ID', 'CELL_LENGTH',
                        'CELL_AREA', 'TOT_CBS', 'FOCI_PER_MICRON', 'CBS_LOCS']
    print(cells_df)
    cells_df.to_csv(cb_file[:-4] + '_Hn_cbcells.csv')

    # save carboxysome properties into different csv
    cbs_df = pd.DataFrame(cbs_quant)
    cbs_df.columns = ['FILE', 'SAMPLE', 'CELL_ID', 'TOT_CBS', 'CB_ID', 'CB_RAD',
                      'CB_AREA', 'CB_TOT_INTENSITY', 'PAIRWISE_DISTANCES', 'CELL_LENGTH']
    print(cbs_df)
    cbs_df.to_csv(cb_file[:-4] + '_cbquant.csv')