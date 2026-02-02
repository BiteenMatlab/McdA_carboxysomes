# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 15:28:04 2024

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
#trace_files = [file for file in filenames if 'avg_traces.csv' in file]
#trace_files = [file for file in filenames if 'avg_traces.csv' in file]

#print(filenames)

'''
columns_avg = [
           '0 nmol avg', '0.005 nmol avg', '0.01 nmol avg', '0.025 nmol avg',
           '0.05 nmol avg', '0.1 nmol avg', '0.125 nmol avg', '0.14 nmol avg',
           ]

columns_stdev = ['0 nmol std', '0.005 nmol std', '0.01 nmol std', '0.025 nmol std', 
                 '0.05 nmol std', '0.1 nmol std', '0.125 nmol std', '0.14 nmol std'
                 ]


columns_avg = [
           '0.005 nmol avg', '0.01 nmol avg', '0.025 nmol avg',
           '0.05 nmol avg', '0.1 nmol avg', '0.125 nmol avg', '0.14 nmol avg',
           ]

columns_stdev = ['0.005 nmol std', '0.01 nmol std', '0.025 nmol std', 
                 '0.05 nmol std', '0.1 nmol std', '0.125 nmol std', '0.14 nmol std'
                 ]
'''

columns_avg = ['McdB 0 nmol', 'McdB 0.01 nmol',	
               'McdB 0.05 nmol', 'McdB 0.1 nmol',
               'McdB 0.25 nmol', 'McdB 0.5 nmol',
               'McdB 0.75 nmol', 'McdB 1 nmol']

colors = ['darkgray', 'darkgreen', 'red', 'blue', 'green', 'violet', 'orange', 'black']



for ii, file in enumerate(filenames[6:]):
    
    print(file)

    df = pd.read_csv(file)
    
    
    fig, axes = plt.subplots(figsize=(3.58,3), dpi=300)

    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    font_size = 12
    
    
    print(len(columns_avg))
    
    for jj, column in enumerate(columns_avg):
        
        
        x = df['Hr'].to_numpy()[:49]
        y = df[column].to_numpy()[:49]
    
        popt, pcov = curve_fit(get_rate, x, y, 
                           maxfev = 1000000)

        plt.plot(x, y,linestyle = 'none', marker='o', markersize=3, color=colors[jj])
        plt.plot(x, get_rate(x, *popt), '--', color='black', linewidth=0.75)
        print(column, popt)
        print(np.sqrt(np.diag(pcov)))
        print()
    

    # Adjust x-axis tick frequency
    plt.xticks([0, .4, .8, 1.2, 1.6, 2, 2.4 ],)  # Ticks at every integer
    plt.xlim(-0.02, 3)
    #plt.ylim(-.5, 10.5)
    plt.ylabel('ATP hydrolyzed (nmol)')
    plt.xlabel('Time (h)')
    
    fig.tight_layout()
    #plt.savefig(directory[:-5] + 'ATPase_plus_DNA.svg', dpi=300) 
    plt.show()