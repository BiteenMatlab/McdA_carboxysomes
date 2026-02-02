# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:47:33 2024

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

def michaelis(substrate, vmax, km): # normal Brownian motion with blurring
    return (vmax*substrate) / (km + substrate)


directory  = sys.argv[1]
filenames = filepull(directory)

fig, axes = plt.subplots(figsize=(3.5,3), dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
print(filenames)

colors = ['black', 'darkgreen']
for ii, file in enumerate(filenames[:]):

    df = pd.read_csv(file)
    print(df)
    
    x_ = df['McdB (uM)'].to_numpy()[:]# * 1000
   
    num_reps = len(df.columns) - 1
    
    y = df['Specific activity'].to_numpy()[:]
    y_stdev = df['Error'].to_numpy()[:]
    y_sem = y_stdev# / np.sqrt(num_reps)
    
    
    #popt, pcov = curve_fit(michaelis, x_, y, maxfev = 10000,sigma=y_sem, absolute_sigma=True)
    
    #print(x_, y_avg, y_sem)
    plt.plot(x_,y,linestyle = 'none', marker='o', markersize=3, color=colors[ii])
    #plt.errorbar(x_, y_avg, y_sem, fmt='none')
    plt.fill_between(x_, y - y_sem, y + y_sem, alpha=0.25, zorder=0, 
                     linewidth=0, color=colors[ii])
    x_fit = np.arange(0, 15, 0.001)
    #plt.plot(x_fit, michaelis(x_fit, *popt), '--', color='black', linewidth=0.75)
    #print(popt)
    
    
    
    # Adjust x-axis tick frequency
plt.xticks([0, 2, 4, 6, 8, 10],)  # Ticks at every integer
plt.xlim(-0.25, 10.5)
plt.ylim(0, 35)
plt.ylabel('Specific activity (1/hr)')
plt.xlabel('McdB (uM)')
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'McdB_dependence.svg', dpi=300) 
plt.show()