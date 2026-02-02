# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 00:13:04 2020

@author: azaldegc
"""


# import modules
import numpy as np
import glob
import sys
import tifffile as tif
import matplotlib.pyplot as plt

from skimage.transform import warp
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from skimage.external import tifffile as sktif


# functions to call
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]

    return filenames


def read_TIFFStack(stack_file, crop=-1):
    '''Reads TIFF stacks, returns stack and number of frames'''

    # stack_file: TIFF stack file name to read
    # -----------------------------------------

    global n_rows, n_cols  # set as global variables to use in other functions

    stack = tif.imread(stack_file)  # read TIFF stack into a np array
    print(stack.shape)
    n_frames = stack.shape[1]  # number of frames in stack

    if crop != -1:
        stack = stack[:crop]
    return stack, n_frames


def save_stack(stack, name):
    '''Save a list of frames as a TIFF stack, returns stack'''
    # stack: list of frame to save as TIFF stack
    # name: filename of original stack
    # -------------------------------------------

    new_stack = np.asarray(stack)  # convert list of new frames to array
    print("new shape", new_stack.shape)
    new_filename = name + '_aligned.tif'
    print(new_filename)
    sktif.imsave(new_filename, new_stack)
    print(new_filename + " saved")

    return new_stack


def pcc_test(original, shifted, pixel='subpixel'):

    # determine image shift based on cross correlation in Fourier space
    # employs upsampled matric-multiplication DFT to achive arbitrary
    # subpixel precision
    shift, error, diffphase = register_translation(original, shifted, 100)

    # initiate figure
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3)

    # original image
    ax1.imshow(original, cmap='gray')
    ax1.set_axis_off()
    ax1.set_title('Reference image')

    # shifted image
    ax2.imshow(shifted.real, cmap='gray')
    ax2.set_axis_off()
    ax2.set_title('Shifted image')

    # calculate cross correlation (pixel or subpixel precision)
    image_product = np.fft.fft2(original) * np.fft.fft2(shifted).conj()
    if pixel == 'subpixel':
        cc_image = _upsampled_dft(
            image_product, 150, 100, (shift*100)+75).conj()
    elif pixel == 'pixel':
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))

    # fourier transform cross correlation
    ax3.imshow(cc_image.real)
    ax3.set_axis_off()
    ax3.set_title("Cross-correlation")

    fig.tight_layout()
    plt.show()
    print(f"Detected pixel offset (y,x): {shift}")

    # -------------------------------------------------------------------------
    # Correcting the shifted image

    # Shift vector
    v, u = shift[0], shift[1]

    # create corrected image
    nr, nc = original.shape[2], original.shape[3]
    row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                         indexing='ij')
    shifted_warp = warp(shifted, np.array([row_coords - v, col_coords - u]),
                        mode='nearest', preserve_range=True)

    # initiate figure
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)

    # shifted image
    ax1.imshow(shifted, cmap='gray')
    ax1.set_title("Unregistered sequence")
    ax1.set_axis_off()

    # corrected image
    ax2.imshow(shifted_warp, cmap='gray')
    ax2.set_title("Registered sequence")
    ax2.set_axis_off()

    # original image
    ax3.imshow(original, cmap='gray')
    ax3.set_title("Target")
    ax3.set_axis_off()

    fig.tight_layout()
    plt.show()


def registration(filename, n, target_stack, stack_01=False, stack_02=False,
                 stack_03=False, stack_04=False, subpixel=True):
    ''' 
    Drift corrects tiff stacks for n channels based on phase contrast channel drift.
    Uses cross-correlation Fourier Transform (ccFT)to determine shift vector, 
    Warps shifted image to correct drift using ccFT shift vector.
        
    n: int, number of frames in stacks
    target_stack: np.array, stack used to determine ccFT shift vector
    stack_01: np.array, RFP stack to be corrected
    stack_02: np.array, YFP stack to be corrected
    stack_03: np.array, DAPI stack to be corrected
    stack_04: np.array, CFP stack to be corrected
    filenames: list, stack filenames
    subpixel: bool, if True then translations are subpixel
        
    '''

    # assign variables for stack names
    target_stack_name = filename[:-4] + '_phase' 
    new_target = []
    print(target_stack_name)
    if type(stack_01) == type(target_stack):
        stack_01_name = filename[:-4] + '_red'
        new_01 = []
    if type(stack_02) == type(target_stack):
        stack_02_name = filename[:-4] + '_mNG'
        new_02 = []
    if type(stack_03) == type(target_stack):
        stack_03_name = filename[:-4] + '_DAPI'
        new_03 = []
    if type(stack_04) == type(target_stack):
        stack_04_name = filename[:-4] + '_mTQ'
        new_04 = []

    # for each frame in stack
    for ii in range(n):

        # if first frame, append to corresponding list
        if ii == 0:

            new_target.append(target_stack[ii])
            if type(stack_01) == type(target_stack):
                new_01.append(stack_01[ii])
            if type(stack_02) == type(target_stack):
                new_02.append(stack_02[ii])
            if type(stack_03) == type(target_stack):
                new_03.append(stack_03[ii])
            if type(stack_04) == type(target_stack):
                new_04.append(stack_04[ii])

            print("frame {} done".format(ii+1))

        # any other following frame
        else:

            # assign variable to reference image and current frame
            original = target_stack[0]
            shifted = target_stack[ii]

            # subpixel precision ccFT or not
            if subpixel:
                shift, error, diffphase = register_translation(original, shifted,
                                                               100)
            elif not subpixel:
                shift, error, diffphase = register_translation(
                    original, shifted)

            # assign variables to shift vector (y, x)
            v, u = shift[0], shift[1]

            # create corrected image array to be filled
            nr, nc = original.shape
            row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                                 indexing='ij')

            # shift reference channel
            shifted_warp_target = warp(shifted,
                                       np.array(
                                           [row_coords - v, col_coords - u]),
                                       mode='nearest', preserve_range=True)
            new_target.append(shifted_warp_target.astype('uint16'))

            # shift mTQ chanel
            if type(stack_01) == type(target_stack):
                shifted_warp_01 = warp(stack_01[ii],
                                       np.array(
                                           [row_coords - v, col_coords - u]),
                                       mode='nearest', preserve_range=True)
                new_01.append(shifted_warp_01.astype('uint16'))

            # shift mNG channel
            if type(stack_02) == type(target_stack):
                shifted_warp_02 = warp(stack_02[ii],
                                       np.array(
                                           [row_coords - v, col_coords - u]),
                                       mode='nearest', preserve_range=True)
                new_02.append(shifted_warp_02.astype('uint16'))

            if type(stack_03) == type(target_stack):
                shifted_warp_03 = warp(stack_03[ii],
                                       np.array(
                                           [row_coords - v, col_coords - u]),
                                       mode='nearest', preserve_range=True)
                new_03.append(shifted_warp_03.astype('uint16'))

            if type(stack_04) == type(target_stack):
                shifted_warp_04 = warp(stack_04[ii],
                                       np.array(
                                           [row_coords - v, col_coords - u]),
                                       mode='nearest', preserve_range=True)
                new_04.append(shifted_warp_04.astype('uint16'))

            print("frame {} done".format(ii+1),
                  "shift (y,x): {}".format(shift))

    # save corrected stacks
    if type(stack_01) == type(target_stack):

        save_stack(new_target, target_stack_name)
        save_stack(new_01, stack_01_name)

    if type(stack_02) == type(target_stack):

        save_stack(new_target, target_stack_name)
        save_stack(new_02, stack_02_name)

    if type(stack_03) == type(target_stack):

        save_stack(new_target, target_stack_name)
        save_stack(new_03, stack_03_name)

    if type(stack_04) == type(target_stack):

        save_stack(new_target, target_stack_name)
        save_stack(new_04, stack_04_name)

    print("Done correcting")


# -----------------------------------------------------------------------------
# algorithm
def main():

    # load data tiff stacks
    foldername = sys.argv[1]
    n_channels = int(sys.argv[2])

    mov_files = filepull(foldername)

    for movie in mov_files:

        if n_channels == 5:
            stack, _ = read_TIFFStack(movie)  # read multichannel tiffstack
            chan1 = stack[0]
            chan2 = stack[1]
            chan3 = stack[2]
            chan4 = stack[3]
            chan5 = stack[4]

            print("Original shape", stack.shape)

            registration(movie, len(chan1[:]), chan1[:],
                         stack_01=chan2[:], stack_02=chan3[:],
                         stack_03=chan4[:], stack_04=chan5[:])

        if n_channels == 4:
            stack, _ = read_TIFFStack(movie)  # read multichannel tiffstack
            chan1 = stack[0]
            chan2 = stack[1]
            chan3 = stack[2]
            chan4 = stack[3]

            print("Original shape", stack.shape)

            registration(movie, len(chan3[:]), chan3[:],
                         stack_01=chan2[:], stack_02=chan3[:], stack_03=chan1[:])

        if n_channels == 3:
            stack, _ = read_TIFFStack(movie)  # read multichannel tiffstack
            chan1 = stack[0]
            chan2 = stack[1]
            chan3 = stack[2]

            print("Original shape", stack.shape)

            registration(movie, len(chan2[:]), chan2[:],
                         stack_01=chan1[:], stack_02=chan3[:])

        if n_channels == 2:
            stack, _ = read_TIFFStack(movie)  # read multichannel tiffstack
            chan1 = stack[0]
            chan2 = stack[1]

            print("Original shape", stack.shape)

            registration(movie, len(chan2[:]), chan2[:],
                         stack_01=chan1[:])

        if n_channels == 1:
            stack, n_frames = read_TIFFStack(movie)
            print("Original shape", stack.shape)

            registration(movie, len(stack[:]), stack[:])

main()
