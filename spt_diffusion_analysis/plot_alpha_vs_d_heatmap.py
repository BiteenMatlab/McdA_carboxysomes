# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:59:42 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from lmfit import Model
from scipy import stats

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def linear(x, m, b):
    return m*x + b

label = '2_vs_100'




directory  = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'alpha' in file]
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
all_data = pd.concat(dfs).reset_index()
print(all_data)
samples = ['Hn_2', 'Hn_14']
colors = ['#199DC5', '#A72428']
print(samples)

fig, axes = plt.subplots(nrows=len(samples), figsize=(2.5, 4), dpi=200) 
ax = axes.ravel()
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9

for ii, sample in enumerate(samples):
    print(sample)
    sample_data = all_data[all_data['Label']==sample]
    
    
    alphas = sample_data['alpha'].to_numpy()#.reshape(-1,1)
    dapps = sample_data['D'].to_numpy()#.reshape(-1,1)
    
    
    xbins = np.linspace(0, 1.5, 25)
    ybins = np.logspace(np.log10(10**-5), np.log10(0.003), 25)
    
    hist, xedges, yedges = np.histogram2d(alphas, dapps, bins=(xbins, ybins))

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",colors[ii]])
    h = ax[ii].hist2d(alphas, dapps, bins=(xbins, ybins),
                       #weights = hist / len(dapps),
                       cmap=cmap)#  density=True)#color=colors[ii], alpha=0.4,edgecolor='none')
    cb = plt.colorbar(h[3], ax = ax[ii])
    cb.outline.set_color('xkcd:black')
    cb.ax.tick_params(labelsize=9)
    ax[ii].set_yscale('log')
    
    ax[ii].set_xlabel('alpha', fontsize=fontsize, ) 
    ax[ii].set_xticks([0, 0.5, 1, 1.5])
    ax[ii].set_ylabel('Diffusion coefficient', fontsize=fontsize)
    
    ax[ii].set_xlim(0, 1.5)
    ax[ii].set_ylim(10**-5, 0.003)
    
    
    
  

#plt.colorbar()

fig.tight_layout()
plt.savefig(directory[:-5] + label + '_alphabyd_heat.svg', dpi=300) 
plt.savefig(directory[:-5] + label + '_alphabyd_heat.png', dpi=300) 
plt.show()