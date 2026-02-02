# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:18:32 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def anomalous_diffusion(tau, D_alpha, alpha):
    return 4 * D_alpha * tau**alpha

name = '2_vs_100_vs_14'
fontsize = 5
directory  = sys.argv[1]
filenames = filepull(directory)

filenames = [file for file in filenames if 'MSD_curves' in file]
print(filenames)

fig, axes = plt.subplots(figsize=(2.95, 2.95), dpi=300) 
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'

#samples = ['Hn_100','Hn_2', 'Hn_14']
colors = ['gray', '#A72428', '#199DC5']


for ii, file in enumerate(filenames[:]):
    
    df = pd.read_csv(file)
    time_lags = df.Time.values[:]
    
    msd_curves = df.to_numpy().T[1:,:]
    print(msd_curves.shape)
    avg_msd = np.mean(msd_curves, axis = 0)
    
    #for msd in msd_curves[:100]:
        #print(msd)    
        #plt.plot(time_lags, msd,alpha=0.25,linewidth=0.5)
       
    
    plt.plot(time_lags, avg_msd, marker='o', color=colors[ii],linestyle='None', 
                 markersize=4, markeredgecolor='black', markeredgewidth=0.5)
    
    #popt, pcov = curve_fit(anomalous_diffusion, time_lags[:9], avg_msd[:9], bounds=(0, [np.inf, 2]))
    #popt = [[1,1],[0.000243, 0.73], [0.000173, 0.56]]
    
    #if ii > 0 :
    #    plt.plot(time_lags[:10], anomalous_diffusion(time_lags[:10], *popt[ii]), '--', color='black')
        
    # Extract fit parameters
    #print(popt)
    sem_msd = np.std(msd_curves, axis = 0) #/ np.sqrt(len(msd_curves))
    #std_msd2 = np.percentile(msd_curves,5, axis = 0)
    #plt.fill_between(time_lags, avg_msd - sem_msd, avg_msd + sem_msd , alpha=0.8, 
    #                linewidth=0,color=colors[ii])
    
    
plt.xlabel('Time Lag (s)')
plt.ylabel('MSD')
plt.xscale('log')
plt.xlim(0.7, 30)
plt.ylim(5*10**-5, 10**-1)
plt.yscale('log')


fig.tight_layout()
plt.savefig(directory[:-5] + name + '_msd_comp.svg', dpi=300) 
plt.savefig(directory[:-5] + name + '_msd_comp.png', dpi=300) 
plt.show()







