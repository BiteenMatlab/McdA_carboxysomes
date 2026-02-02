# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 12:02:48 2025

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
files = [file for file in files if 'intensities.csv' in file]

dfs = [pd.read_csv(file) for file in files]

data = pd.concat(dfs)
#print(data)

intensities = data['AVERAGE_INTENSITY'].to_numpy()
print("number of cells", len(intensities))

binwidth = 1000
binBoundaries = np.arange(min(intensities),
                          max(intensities), binwidth)

params_1 = (1,1250, 500)
hist, bin_edges = np.histogram(intensities, bins=binBoundaries, 
                               weights = np.zeros_like(intensities) +
                               1 / intensities.size)# density=True)  
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
x, bins, p = plt.hist(intensities,  color='gray',
                     edgecolor='k', bins=binBoundaries,
                     weights = np.zeros_like(intensities) + 1 / intensities.size)

print(result.params['cen1'].value, result.params['sigma1'].value)
x_smooth = np.linspace(min(intensities),
                         max(intensities), 1000)
yfit = gaussian(x_smooth, result.params['amp1'].value ,result.params['cen1'].value, result.params['sigma1'].value)
plt.plot(x_smooth, yfit, 'k--', label='Gaussian Fit')
plt.xlabel('single cell intensity')
plt.ylabel('Normalized Frequency')
#plt.xlim(0,35000)
#plt.ylim(0, 0.2)
#plt.xticks([0, 10000, 20000, 30000])
#plt.title('Confinement Radius Histogram')
fig.tight_layout()
#plt.savefig(directory[:-5] + name + '_uncertainty.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_uncertainty.png', dpi=300)
plt.show()
