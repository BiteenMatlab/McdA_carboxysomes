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
    
    x_ = df['ATP (mM)'].to_numpy()[:-2]
   
    num_reps = len(df.columns) - 1
    
    y = []
    
    for rep in range(num_reps):
        
        # correct for units in concentration and time
        data = (df.iloc[:,rep+1].to_numpy()[:-2] / .1) * 60
        y.append(data)
    
    
    y_ = np.asarray(y)
    
    y_avg = np.average(y_, axis=0)
    y_stdev = np.std(y_, axis=0)
    y_sem = y_stdev / np.sqrt(num_reps)
    
    
    popt, pcov = curve_fit(michaelis, x_, y_avg, 
                           maxfev = 10000,
                           sigma=y_sem, absolute_sigma=True)
    
    #print(x_, y_avg, y_sem)
    plt.plot(x_,y_avg,linestyle = 'none', marker='o', markersize=3, color=colors[ii])
    #plt.errorbar(x_, y_avg, y_sem, fmt='none')
    plt.fill_between(x_, y_avg - y_sem, y_avg + y_sem, alpha=0.25, zorder=0, 
                     linewidth=0, color=colors[ii])
    x_fit = np.arange(0, 2.21, 0.001)
    plt.plot(x_fit, michaelis(x_fit, *popt), '--', color='black', linewidth=0.75)
    print(popt)
    

plt.ylabel('Specific activity (1/hr)')
plt.xlabel('ATP (mM)')
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'ATP_dependence.svg', dpi=300) 
plt.show()