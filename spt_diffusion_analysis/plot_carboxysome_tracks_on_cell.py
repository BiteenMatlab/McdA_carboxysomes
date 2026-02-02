# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 14:13:47 2022

@author: azaldegc


Script make single cell trajectory images:
    
    Needs a directory with the tracks.csv and original phase image .tif 
    and phase mask .tif file
    
"""


# import modules
import sys
import numpy as np
import glob
import seaborn as sns
import tifffile as tif

import pandas as pd
import matplotlib.pyplot as plt


from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

from matplotlib import cm
from matplotlib.colors import Normalize 
from scipy.interpolate import interpn
from matplotlib_scalebar.scalebar import ScaleBar
from math import log
from scipy.optimize import curve_fit
import matplotlib.collections as mcoll
import matplotlib.path as mpath

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# read csv
def csv2df(file):
    '''
    Reads a .csv file as a pandas dataframe
    
    input
    -----
    file: string, csv filename
    
    output
    -----
    df: pd.dataframe
    '''
    df = pd.read_csv(file, header = 0)
    return df

def total_tracks(df):
    n_tracks = df['TRACK_ID'].max()
   
    return int(n_tracks)

# Separate tracks by ID into dictionary
def tracks_df2dict(df, n_tracks, min_length):
    
    tracks_dict = {} # initialize empty dictionary
    
    for ii in range(n_tracks): # for each track
        
        track = df.loc[df.TRACK_ID == (ii)] # assign current track to var
        
        track_length = len(track) # find track length (n localizations)
        gap = has_gap(track["FRAME"].tolist())
        # if track length > or = min_length then store in dict
        if track_length >= min_length and gap == False: 
            tracks_dict["track_{0}".format(ii)] = df.loc[df.TRACK_ID == (ii)]
                   

    return tracks_dict

def has_gap(values):
    sorted_values = values
            
    for i in range(len(sorted_values) - 1):
        if sorted_values[i+1] - sorted_values[i] > 4:
            return True
    return False

# calculate MSD curves and apparent diffusion coefficient
def calc_MSD(track, tracks_dict, req_locs, plot=False):
    # corresponding column number for x and y coordinates    
    x_coord = 4
    y_coord = 5
    # determine # of localizations of track
    track_n_locs = len(tracks_dict[track])
    # determine # of steps of track
    track_n_steps = track_n_locs - 1 #req_steps
    # hold list for MSD of a track
    MSD_track = []        
    # start time_lag
    time_lag = 0
    # run a lag calc until max step num
    for lag in range(max_tau): # changed the -1
        # keep count of time lag
        time_lag += 1
        # num of steps to calc
        n_steps = track_n_steps - time_lag
        #global displacement_list
        sq_displacement_list =[]
        # for each step of time lag
        for j, step in enumerate(range(n_steps)):
            x0, xF = loc_in_dict(tracks_dict, track, j, x_coord,time_lag)
            y0, yF = loc_in_dict(tracks_dict, track, j, y_coord,time_lag)
            sq_displacement = (calc_stepsize(x0,xF,y0,yF))**2
            sq_displacement_list.append(sq_displacement)
        mean_sq_dis = np.average(sq_displacement_list)
        MSD_track.append(mean_sq_dis)
    
    # from MSD curves, estimate diffusion coefficient
    tau_list = [(x*framerate) + framerate for x in range(len(MSD_track))]  
    ydata = np.asarray(MSD_track[:max_tau])
    xdata = np.asarray(tau_list[:max_tau])
    diff, _ = curve_fit(brownian_a, xdata, ydata,
                            maxfev = 100000)

    # return diffusion coefficient
    return diff[0], diff[1],         

# calculate size of step
def calc_stepsize(x0,xF,y0,yF):
    x_step = xF - x0
    y_step = yF - y0
    stepsize_pix = np.sqrt(x_step**2 + y_step**2)
    stepsize_um = stepsize_pix*pixsize 
    return stepsize_um


# determine position in dictionary
def loc_in_dict(data,key,row,col,time_lag):
    loc0 = data[key].iloc[row,col]
    locF = data[key].iloc[row+time_lag,col]    
    return loc0,locF

def brownian_a(tau,D,a): # normal Brownian motion
    return 4*D*tau**a 

def brownian(tau,D): # normal Brownian motion
    return 4*D*(tau)

def brownian_blur_corr(tau,D, sig): # normal Brownian motion with blurring
    return (8/3)*D*(tau) + 4*sig**2



def colorline(
    x, y, z=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc


def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def main():
    
    # set global parameters
    global pixsize, min_locs, minstepsize, max_tau, framerate
    framerate = 1 # frame acquisition every x seconds
    pixsize = 0.066 # microns per pixel 
    minstepsize = 0.00
    min_track_len = 10# in localizations
    max_tau = 8
    saveplot = True
    showplot = True

    # load fits csvs and phase image tiffs
    directory = sys.argv[1]
    files = filepull(directory)
    
    phaseImg_files = [file for file in files if ".tif" in file and "PhaseMask" not in file]
    tracks_files = [file for file in files if "spots.csv" in file]
 
      
    tracks_files.sort()
    phaseImg_files.sort()
    
    
    # for each image and fit
    for ii,file in enumerate(tracks_files[:]):
        
        # read fits csv
        tracks_df = csv2df(file)
        # convert tracks df to dictionary
        dataset = tracks_df2dict(tracks_df, total_tracks(tracks_df), 
                                 min_track_len)
                
        
        
        # read phase img
        phaseImg = tif.imread(phaseImg_files[ii])
    #    phasemask = tif.imread(mask_files[ii])
        blankimg = np.zeros((phaseImg.shape[0],phaseImg.shape[1]), 
                            np.float64)
        phaseImg_blur = filters.gaussian(phaseImg,0.5) 
        # initiate image
        fig,ax = plt.subplots(dpi=300)
        # plt.rcParams.update({'font.size': 9})
        plt.rcParams['font.family'] = 'Calibri'
        sns.set(font="Calibri")
        plt.rcParams['svg.fonttype'] = 'none'
        font_size = 12
  
        plt.yticks(fontsize=font_size)
        plt.xticks(fontsize=font_size)   
        
        
        numrows, numcols = blankimg.shape
        #plt.axis('off')
        plt.imshow(phaseImg_blur, cmap='gray', zorder=0)
     
        
        from matplotlib import colors
        
        
        
        norm = colors.Normalize(vmin=0, vmax=25)
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        cmap.set_array([])
        
        for track in dataset:
            
            #D, alpha, = calc_MSD(track, dataset, min_track_len)
            if 1 > 0:
                
                x = dataset[track].iloc[:,4].to_numpy()
                y = dataset[track].iloc[:,5].to_numpy()

             
                sc = ax.plot(dataset[track].iloc[:,4], dataset[track].iloc[:,5],
                             linewidth=.25, alpha=1)

           
        #cbar = fig.colorbar(cmap)
        #cbar.outline.set_color('xkcd:black')
        #cbar.ax.set_ylabel('Time (s)', fontsize=font_size)
       # tick_font_size = font_size
        #cbar.ax.tick_params(labelsize=font_size)                                                   
        plt.ylim(numrows, 0)
        '''
        scalebar = ScaleBar(pixsize, 'um', color = 'k', box_color = 'None',
                            location = 'upper right', fixed_value = 1, scale_loc='none',
                            width_fraction=0.025, label_loc='none', label=None
                            ) 
        ax.add_artist(scalebar)
        '''
        
        fig.tight_layout()
        if saveplot == True:
            plt.savefig(phaseImg_files[ii][:-4]+"_trackMap_v03.png", 
                    dpi=400, 
                    bbox_inches='tight')
            plt.savefig(phaseImg_files[ii][:-4]+"_trackMap_v03.svg", 
                    dpi=400, 
                    bbox_inches='tight')

        if showplot == True:
            plt.show()
        
main()    