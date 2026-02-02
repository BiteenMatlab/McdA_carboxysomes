# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 20:59:47 2024

@author: azaldegc
"""

import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory  =sys.argv[1]
filenames = filepull(directory)
filenames.sort()

fig, axes = plt.subplots(figsize=(4.5,3.5), dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
print(filenames)

colors = ['darkgray', 'darkgreen', 'blue', 'red', 'magenta']
for ii, file in enumerate(filenames[:]):

    df = pd.read_csv(file)
    print(df)
    
    time = df['Lifetime'].to_numpy()
   
    counts = df['Survival Probability'].to_numpy()
    
    fit = df['Fit'].to_numpy()
    
   
    plt.plot(time, fit, '--', color='black', linewidth=0.75)
    plt.plot(time, counts, linestyle = 'none', marker='o', markersize=3, color=colors[ii])
    
    
    
    
    
    
    # Adjust x-axis tick frequency
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.ylim(-100, 4800)
plt.xlim(-0.1, 12)
plt.xticks([0, 2, 4, 6, 8, 10, 12])
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'lifetime_plot.svg', dpi=300) 
plt.show()