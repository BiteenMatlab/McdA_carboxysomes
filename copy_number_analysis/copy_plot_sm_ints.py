# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 18:17:51 2025

@author: azaldegc
"""

import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lmfit import Model


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))

# load directory with files
directory = sys.argv[1]
files = filepull(directory)
trace_files = [file for file in files if 'intensities.csv' in file]

trace_dfs = [pd.read_csv(file) for file in trace_files]

traces = pd.concat(trace_dfs)
traces.columns = ['index', 'sm_ints']
print(traces)

data = traces['sm_ints'].to_numpy()
print(len(data))
binwidth = 2500
binBoundaries = np.arange(min(data),
                          max(data), binwidth)

params_1 = (1,12000, 5000)
hist, bin_edges = np.histogram(data, bins=binBoundaries, 
                               weights = np.zeros_like(data) +
                               1 / data.size)# density=True)  
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
model = Model(gaussian)
params = model.make_params(amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])

result = model.fit(hist, x=bin_centers, params=params)

fig, axes = plt.subplots(figsize=(2.75,2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
x, bins, p = plt.hist(data,  color='gray',
                     edgecolor='k', bins=binBoundaries,
                     weights = np.zeros_like(data) + 1 / data.size)

print(result.params['cen1'].value, result.params['sigma1'].value)
x_smooth = np.linspace(min(data),
                         max(data), 1000)
yfit = gaussian(x_smooth, result.params['amp1'].value ,result.params['cen1'].value, result.params['sigma1'].value)
plt.plot(x_smooth, yfit, 'k--', label='Gaussian Fit')
plt.xlabel('Magnitude of photobleaching events ')
plt.ylabel('Normalized Frequency')
plt.xlim(0,35000)
#plt.ylim(0, 0.2)
plt.xticks([0, 10000, 20000, 30000])
#plt.title('Confinement Radius Histogram')
fig.tight_layout()
#plt.savefig(directory[:-5] + name + '_uncertainty.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_uncertainty.png', dpi=300)
plt.show()

