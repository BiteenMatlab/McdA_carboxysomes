# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 21:58:32 2024

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

def get_kcat(conc, kcat, v0): # normal Brownian motion with blurring
    return kcat*conc + v0


directory  =sys.argv[1]
filenames = filepull(directory)

fig, axes = plt.subplots(figsize=(3.58,3), dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
print(filenames)

colors = ['black', 'black', 'darkgray', 'darkgreen',  ]
for ii, file in enumerate(filenames[3:]):

    df = pd.read_csv(file)
    print(df)
    
    # correct for units in concentration
    x_ = df['McdA (uM)'].to_numpy()*0.1
   
    num_reps = len(df.columns) - 1
    
    y = []
    
    for rep in range(num_reps):
        
        data = df.iloc[:,rep+1].to_numpy()
        y.append(data)
    
    
    y_ = np.asarray(y)
    
    y_avg = np.average(y_, axis=0)
    y_stdev = np.std(y_, axis=0)
    y_sem = y_stdev / np.sqrt(num_reps)
    
    
    #popt, pcov = curve_fit(get_kcat, x_, y_avg, 
    #                       maxfev = 10000,)
                           #sigma=y_sem, absolute_sigma=True)
    
    #print(x_, y_avg, y_sem)
    plt.plot(x_,y_avg,linestyle = 'none', marker='o', markersize=3, color=colors[ii])
    #plt.errorbar(x_, y_avg, y_sem, fmt='none')
    plt.fill_between(x_, y_avg - y_sem, y_avg + y_sem, alpha=0.5, zorder=0, 
                     linewidth=0, color=colors[ii])
   # plt.plot(x_, get_kcat(x_, *popt), '--', color='black', linewidth=0.75)
    #print(popt)
    
    
    
    # Adjust x-axis tick frequency
plt.xticks([0, .2, .4, .6, .8, 1.0, 1.2, 1.4, ],)  # Ticks at every integer
plt.xlim(-0.02, 1.5)
plt.ylim(-1, 5.5)
plt.ylabel('ATP hydrolyzed (nmol)')
plt.xlabel('SUMO-McdA (uM)')
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'McdB_ATPase_plus_DNA.svg', dpi=300) 
plt.show()