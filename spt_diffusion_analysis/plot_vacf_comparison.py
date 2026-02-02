# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 15:52:43 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from lmfit import Model

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames



name = '2_vs_14'
fontsize = 5
directory  = sys.argv[1]
filenames = filepull(directory)

filenames = [file for file in filenames if 'vacf_norm' in file]
print(filenames)
'''
fig, axes = plt.subplots(figsize=(3.75,3), dpi=300) 
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'

samples = ['Hn_14', 'Hn_2']
colors = ['#A72428','#199DC5', ]
plt.axvline(x=1, linestyle='dashed', color='gray')
plt.axhline(y=0, linestyle='dashed', color='gray')
'''
vacf1 = []
for ii, file in enumerate(filenames[:]):
    
    df = pd.read_csv(file)
    time_lags = df.Time.values[:]
    
    vacf_curves = df.to_numpy().T[1:,:]
    print(vacf_curves.shape)
    avg_vacf = np.mean(vacf_curves, axis = 0)
    print(len(vacf_curves))
    
    #for msd in msd_curves[:100]:
        #print(msd)    
        #plt.plot(time_lags, msd,alpha=0.25,linewidth=0.5)
       
    
   # plt.plot(time_lags, avg_vacf, color=colors[ii],  linewidth=1, marker='o', markersize=3)
    
    
    VACF_t1 = np.asarray([item[1] for item in vacf_curves])
    vacf1.append(VACF_t1)

    avg_vacf1 = np.mean(VACF_t1, axis = 0)
    std_vacf1 = np.std(VACF_t1, axis = 0) / np.sqrt(len(VACF_t1))
    percentile_vacf1 = np.percentile(VACF_t1,95, axis = 0)
    print("tau 1: ", avg_vacf1, std_vacf1)
   
'''
plt.xlabel('Time Lag (s)')
plt.ylabel('Velocity Autocorrelation')
plt.xticks([0, 2, 4, 6, 8])
plt.xlim(0, 8.5)
plt.ylim(-.4,1.1)
fig.tight_layout()
plt.savefig(directory[:-5] + name + '_vacf_comp.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_vacf_comp.png', dpi=300) 
plt.show()
'''

def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))
    


fig, axes = plt.subplots(figsize=(2.75,2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
samples = ['Hn_14', 'Hn_2']
colors = ['#A72428','#199DC5', ]
print(samples)
for ii, data in enumerate(vacf1):
   
    
    binwidth = 0.05
    binBoundaries = np.arange(min(data),max(data), binwidth)
    #x, bins, p = plt.hist(data, bins=binBoundaries, color=colors[ii],
    #                      histtype = 'step',
    #                     edgecolor=colors[ii],   weights = np.zeros_like(data) + 1 / data.size)#density=True)
    # plot histogram
    histogram, bins = np.histogram(data, bins=binBoundaries, weights = np.zeros_like(data) + 1 / data.size)
    normalized_histogram = histogram / np.sum(histogram)
    plt.plot(bins[:-1], normalized_histogram, marker='o', ls='-',
             linewidth=1, markersize=3, color=colors[ii])
    
   
    params_1 = (100,-.25, .1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    model = Model(gaussian)
    params = model.make_params(amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])
    result = model.fit(histogram, x=bin_centers, params=params)

    print(result.params['cen1'].value, result.params['sigma1'].value)

    #x, bins, p = plt.hist(data, bins=binBoundaries, color=colors[ii],
    #                     edgecolor='k', weights = np.zeros_like(data) + 1 / data.size)# density=True)
    
    #plt.plot(xx, yy, color='red')
  
    
 
   

plt.xlabel('Cv (t=1 s)', fontsize=fontsize, ) 
plt.ylabel('Fraction of tracks', fontsize=fontsize)
#plt.xscale('log')
plt.xlim(-.8, 0.45)
plt.xticks([-.8, -.4, 0, .4])
plt.ylim(0,.2)
fig.tight_layout()
plt.savefig(directory[:-5] + name + 'cvt.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_cvt.png', dpi=300) 
plt.show()

