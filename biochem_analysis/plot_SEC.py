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
names = ['R158E']

print(len(filenames))
for ii, file in enumerate(filenames[2:]):
    fig, axes = plt.subplots(figsize=(4,2.25), dpi=300)

    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    font_size = 12
    print(filenames)

    colors = ['black', 'darkgreen']

    df = pd.read_csv(file)
    print(df)
    
    x = df['mL'].to_numpy()[:]
   
    #num_reps = len(df.columns) - 1
    
    
    
    data_plus_atp = df['SUMO-Hn_McdA_ATP'].to_numpy()#[1770:]
    data_minus_atp = df['SUMO-Hn_McdA'].to_numpy()#[1770:]
        

    plt.plot(x, data_minus_atp, linestyle = '-', color='black')
    plt.plot(x, data_plus_atp, linestyle = '-', color='forestgreen')


    # Adjust x-axis tick frequency
    
    plt.xticks([2, 2.2, 2.4, 2.6, 2.8, 3, 3.2],)
    plt.yticks([0, 5, 10, 15, 20, 25],)# Ticks at every integer
    plt.xlim(2, 3.2)
    plt.ylim(-1, 25)
    plt.ylabel('Absorbance 280 nm')
    plt.xlabel('Elution (mL)')
    
    fig.tight_layout()
    plt.savefig(directory[:-5] + names[ii] + '_SEC.svg', dpi=300) 
    plt.show()