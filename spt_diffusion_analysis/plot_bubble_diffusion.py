# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 12:18:20 2024

@author: azaldegc
"""
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt


# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


directory  =sys.argv[1]
filenames = filepull(directory)


for file in filenames[-1:]:

    df = pd.read_csv(file)
    
    x = df['Sample'].to_numpy()
    print(x)
    y = df['Diffusion Coefficient']
    y_err = df['Error_D']
    x = [1, 1, 2, 2, 3, 3, 4, 4]
    
    w = df['Weight']
    
    bubble_size = 1000
    fig, axes = plt.subplots(figsize=(3,2), dpi=300)

    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    font_size = 12
    plt.axhline(y=0.042, linestyle='dashed', color='gray', linewidth=0.75, zorder=0)
    plt.scatter(x=x, y=y , s=w*bubble_size)
    plt.errorbar(x=x, y=y, yerr=y_err, fmt='none', color='white', 
                 capsize=2, capthick=0.5, linewidth=0.5)
    # Adjust x-axis tick frequency
    plt.xticks([0.5,1,2,3,4,4.5],)  # Ticks at every integer
    plt.ylim(0.001, 5)
    plt.yscale('log')
    plt.ylabel('diffusion coefficient')
    
    fig.tight_layout()
    plt.savefig(directory[:-5] + 'bubble.svg', dpi=300) 
    plt.show()

    
    
    
    

