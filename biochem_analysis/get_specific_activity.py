# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:28:18 2024

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

def get_rate(t, r, b): 
    # t, ATP at time point
    # r, rate; ATP turnover per unit time
    # b, intercept
    return r*t + b


directory  = sys.argv[1]
filenames = filepull(directory)
trace_files = [file for file in filenames if 'DNA.csv' in file]

print(trace_files)


labels = ['McdA_plus_DNA', 
          'McdA_no_DNA',
          'McdA_K20A_no_DNA','McdA_K20A_plus_DNA', 
          'McdA_R158E_no_DNA','McdA_R158E_plus_DNA']



colors = ['darkgray', 'darkgreen', 'red', 'blue', 'green', 'violet', 'orange', 'black']

for ii, file in enumerate(filenames[:]):
    
    print(file)
    label = labels[ii]

    df = pd.read_csv(file)
    
    
    fig, axes = plt.subplots(figsize=(3.6,3), dpi=300)

    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    font_size = 12
    print(file)
    
    # define axis
    x = df['McdA nmol'].to_numpy()
    y = df['ATP per hour'].to_numpy()[:]
    std = df['error'].to_numpy()[:]
    
    # fitting routine
    popt, pcov = curve_fit(get_rate, x[:], y[:], 
                           maxfev = 1000000,
                           sigma=std[:], absolute_sigma=True)
    # plot curves        
    plt.plot(x, y,linestyle = 'none', marker='o', markersize=3, color=colors[ii])        
    plt.fill_between(x, y - std, y + std, alpha=0.5, zorder=0, 
                     linewidth=0, color=colors[ii])
    x_ = np.linspace(0, 1, 50)
    plt.plot(x_, get_rate(x_, *popt), '--', color='black', linewidth=0.75)
    
    # print fit params
    print(label, popt)
    print(np.sqrt(np.diag(pcov)))
    print()
    

    # Adjust x-axis tick frequency
    plt.xticks([0, 0.03, .06, .09, .12, .15],)  # Ticks at every integer
    plt.xlim(0, .15)
    plt.ylim(-.05, 5)
    plt.ylabel('ADP production rate (nmol/h)')
    plt.xlabel('[McdA] nmol')
    
    fig.tight_layout()
    plt.savefig(directory[:-5] + label + 'specific activity.svg', dpi=300) 
    plt.show()