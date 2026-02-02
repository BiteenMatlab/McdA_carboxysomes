# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 18:58:31 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))

label = '2_vs_100'




directory  = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'nucleoid_analysis' in file]
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
all_data = pd.concat(dfs).reset_index()
print(all_data)

fig, axes = plt.subplots(figsize=(3., 3), dpi=150) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
samples = ['20241121_Hn2']#'Hn_100']
colors = ['gray']
print(samples)
for ii, sample in enumerate(samples):
    print(sample)
    sample_data = all_data[all_data['Sample']==sample]
    
    
    data = sample_data['Nucleoid_Length'].to_numpy().reshape(-1,1)
    data = data[~np.isnan(data)]
    print(len(data))
    #print(data)
    print("mean + stdev", np.mean(data), np.std(data))

    #params_1 = (100,2.5, 1)
    binwidth = 0.05
    binBoundaries = np.arange(min(data),max(data), binwidth)
    hist, bin_edges = np.histogram(data, bins=binBoundaries, weights = np.zeros_like(data) + 1 / data.size)#density=True)   
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    #model = Model(gaussian)
    #params = model.make_params(amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])
    #result = model.fit(hist, x=bin_centers, params=params)

    #print(result.params['cen1'].value, result.params['sigma1'].value)

    x, bins, p = plt.hist(data, bins=binBoundaries, color=colors[ii],
                         edgecolor='k', weights = np.zeros_like(data) + 1 / data.size)# density=True)
   # plt.plot(bin_centers, result.best_fit, 'k-', label='Gaussian Fit')


plt.xlabel('Cell length (um)', fontsize=fontsize, ) 
plt.ylabel('Fraction of cells', fontsize=fontsize)
plt.yscale('linear')
#plt.xticks([.5, 1, 1.5,2, 2.5])
#plt.yticks([0, 0.05,0.1])
#plt.ylim(0, 0.13)
#plt.xlim(0, 5)
fig.tight_layout()
plt.savefig(directory[:-5] + label + '_nuc_len.svg', dpi=300) 
plt.savefig(directory[:-5] + label + '_nuc_len.png', dpi=300) 
plt.show()