# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 17:08:02 2024

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


def gaussian(x,  cen1, sigma1):
        return np.exp(-(x - cen1)**2 / (2 * sigma1**2))

label = '2_vs_100'

n_bins = 15


directory  = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'radius_confinements' in file]
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
all_data = pd.concat(dfs).reset_index()
print(all_data)

fig, axes = plt.subplots(figsize=(2.7, 2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
samples = [ 'Hn_431_80_TL220ms','Hn_441_80_TL220ms', 'Hn_14_80_TL220ms']
colors = ['green', 'mediumorchid', 'gray']
print(samples)
plt.axvline(x=31, linestyle='dashed', color='gray', linewidth=0.75)
for ii, sample in enumerate(samples):
    print(sample)
    sample_data = all_data[all_data['name']==sample]
    
    
    data = sample_data['Radius of Confinement'].to_numpy()*1000#.reshape(-1,1)
    print(len(data), np.mean(data))
    binBoundaries = np.arange(5,max(data),15)
    bootstraps = 10
    means = []
    histograms = []
    for n in range(bootstraps):
        bootstrapped_data =  np.random.choice(data, replace=True, 
                                 size=int(len(data)))
        means.append(np.mean(bootstrapped_data))
    
        histogram, bins = np.histogram(bootstrapped_data, bins=binBoundaries)
        normalized_histogram = histogram / np.sum(histogram)
        histograms.append(normalized_histogram)
        
    #histograms = np.asarray(histograms)
    histogram_avg =np.mean(np.asarray(histograms), axis=0)
    histogram_CIlow = np.percentile(np.asarray(histograms), 5, axis=0)
    histogram_CIhigh = np.percentile(np.asarray(histograms), 95, axis=0)
    print("bootstrapped mean: ", np.mean(means), np.std(means))
    
    plt.fill_between(bins[:-1], histogram_CIlow, histogram_CIhigh, 
                       color=colors[ii], alpha=0.5)
    plt.plot(bins[:-1], histogram_avg, ls='-', linewidth=0.5,
             marker='o',markersize=2,
             color=colors[ii])
    


plt.xlabel('Confinement radius (um)', fontsize=fontsize, ) 
plt.ylabel('Fraction of tracks', fontsize=fontsize)
#plt.xscale('log')
plt.xlim(0, 200)
plt.xticks([0, 50, 100, 150, 200])
plt.ylim(-0.01, 0.85)



fig.tight_layout()
plt.savefig(directory[:-5] + label + '_confinement_radius.svg', dpi=300) 
plt.savefig(directory[:-5] + label + '_confinement_radius.png', dpi=300) 
plt.show()