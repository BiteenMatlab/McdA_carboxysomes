# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:10:23 2025

@author: azaldegc
"""

import sys
import numpy as np
import glob
import tifffile as tif
import matplotlib.pyplot as plt
from skimage import (feature, filters)
from scipy.signal import savgol_filter
import pandas as pd


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# load directory with files
directory = sys.argv[1]
files = filepull(directory)
trace_files = [file for file in files if '_traces.csv' in file]




for file in trace_files:
    
    df = pd.read_csv(file)
    num_columns = df.shape[1]
    
    intensity_diff_arr = []

    for i in range(num_columns)[1:]:
       
        trace = df.iloc[:, i].to_numpy()  # Access columns by index(file)
        #print(trace)
        
        plt.plot(range(len(trace)), trace)
        changepoints = []
        plt.show()
        
        # Prompt the user to enter values
        user_input = input("Enter a list of values, separated by commas: ")

        # Split the input string into a list
        values_list = user_input.split(',')

        # Optionally, convert the elements to a specific type (e.g., integers)
        values_list = [int(value.strip()) for value in values_list]
    
        print("You entered:", values_list)
        if 0 not in values_list:
            for value in values_list:
            
                intensity_start = np.average(trace[value - 7: value])
                intensity_end = np.average(trace[value: value + 7])
                intensity_diff = intensity_start - intensity_end
                intensity_diff_arr.append(abs(intensity_diff))
                
                print(intensity_diff_arr)
        elif 0 in values_list:
            continue
            
    intensities = np.asarray(intensity_diff_arr).T   
    data_df = pd.DataFrame(intensities)
    #print(data_df)
    data_df.to_csv(file[:-4] + '_singlemolecule_intensities.csv')   
    print("Done with file: ", file)      
           
    