# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 21:13:51 2024

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

def get_k_diss(k_tl, k_diss, k_bleach_tl): # normal Brownian motion with blurring
    return k_diss*k_tl + k_bleach_tl


directory  =sys.argv[1]
filenames = filepull(directory)

fig, axes = plt.subplots(figsize=(2.5,2.5), dpi=300)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
print(filenames)

colors = ['darkgray', 'darkgreen']
for ii, file in enumerate(filenames[:]):

    df = pd.read_csv(file)
    print(df)
    
    tau = df['tau'].to_numpy()
    dissociation_rate = df['Kapp x Ttl'].to_numpy()
    error =  df['error bars'].to_numpy()
   
    
    
    popt, pcov = curve_fit(get_k_diss, tau, dissociation_rate, 
                           maxfev = 10000,
                           sigma=error, absolute_sigma=True)
    
    #print(x_, y_avg, y_sem)
    plt.plot(tau, dissociation_rate,linestyle = 'none', marker='o', markersize=3, color=colors[ii])
    
    plt.fill_between(tau, dissociation_rate - error, dissociation_rate + error, alpha=0.5, zorder=0, 
                     linewidth=0, color=colors[ii])
    plt.plot(tau, get_k_diss(tau, *popt), '--', color='black', linewidth=0.75)
    print(popt)
    print('error', np.sqrt(np.diag(pcov)))
    
    
    # Adjust x-axis tick frequency
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1 ],) 
plt.yticks([0, 0.1, 0.2, 0.3, 0.4],)  # Ticks at every integer
plt.xlim(0, 1.05)
plt.ylim(0, .45)
plt.ylabel('k_app x TL')
plt.xlabel('t_TL')
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'true_k_diss.svg', dpi=300) 
plt.show()