# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 23:21:18 2024

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

def onesite_bind_isotherm(ligand, bmax, Kd, bg): # normal Brownian motion with blurring
    return (bmax*ligand) / (Kd + ligand) + bg #+ 40

def hyperbinding(protein, p0, pmax, Kd):
    return p0 + (pmax - p0)*(protein / (Kd + protein))


directory  = sys.argv[1]
filenames = filepull(directory)

fig, axes = plt.subplots(figsize=(3.58,3), dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
print(filenames)

colors = ['darkgreen', 'midnightblue', 'steelblue', 'black', 'gray']
#colors = ['gray', 'darkgreen']
for ii, file in enumerate(filenames[:5]):
    
    print(file)

    df = pd.read_csv(file)
    #print(df)
    
    x_ = df['Protein [uM]'].to_numpy()[:]
   # print(x_)
   
    num_reps = len(df.columns) - 1
    
    y = []
    
    for rep in range(num_reps):
        
        data = df.iloc[:,rep+1].to_numpy()[:]
        y.append(data)
        
        #popt, pcov = curve_fit(onesite_bind_isotherm, x_, data, 
        #                       maxfev = 100000, bounds = (0, [np.inf, np.inf, np.inf]))
        #                       sigma=y_sem, absolute_sigma=True)
        #print(popt)
        #print(np.sqrt(np.diag(pcov)))
                           
                              
    y_ = np.asarray(y)
    
    y_avg = np.average(y_, axis=0)
    y_stdev = np.std(y_, axis=0)
    y_sem = y_stdev #/ np.sqrt(num_reps)
    
    popt, pcov = curve_fit(hyperbinding, x_, y_avg,
                           bounds = (0, [100, 100, 1000]),
                           maxfev = 100000,
                           sigma=y_stdev, absolute_sigma=True)
    
    
    #print(x_, y_avg, y_sem)
    plt.plot(x_[1:],y_avg[1:],linestyle = 'none', marker='o', markersize=2, color=colors[ii])
    #plt.errorbar(x_, y_avg, y_sem, fmt='none')
    plt.fill_between(x_[1:], y_avg[1:] - y_sem[1:], y_avg[1:] + y_sem[1:], alpha=0.25, zorder=0, 
                     linewidth=0, color=colors[ii])
    x_fit = np.arange(0.01, 45, 0.001)
    plt.plot(x_fit, hyperbinding(x_fit, *popt), '--', color=colors[ii], linewidth=0.75)
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    print()
    
    
    
    # Adjust x-axis tick frequency
#plt.xticks([0, 10, 20, 30, 40],)  # Ticks at every integer
plt.yticks([10, 20, 30, 40, 50, 60, 70, 80],) 
plt.xlim(0.018, 45)
plt.xscale('log')
plt.ylim(10, 88)
plt.ylabel('Fluorescence polarization (mP)')
plt.xlabel('McdA [uM]')
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'polarization_hyper_.svg', dpi=300) 
plt.show()