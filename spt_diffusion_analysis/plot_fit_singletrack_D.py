# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 17:33:27 2022

@author: azaldegc
"""

import matplotlib.pyplot as plt
import matplotlib as mlab
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
from sklearn.mixture import GaussianMixture
import scipy.stats as stats
from scipy.optimize import curve_fit

# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

# user defined parameters
fit = 'Mix' # gaussian mixture, do not change
n_comps = 1# number of gaussian states to fit to 
bootstraps = 100 # number of boostraps to perform 
boots_percent = 1 # fraction of data to sample
n_bins = 20 # number of bins, for plotting
#font_size = 25 # fontsize, for plotting
plot = True # for plotting results
common_norm_ = False
n_samples = 5


# data organization
directory  =sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'Dcoeff' in file]
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
    #samples.append(df['Label'].iloc[0])
all_data = pd.concat(dfs).reset_index()
#samples = list(set(samples))
#samples.reverse()
#samples.sort()
samples = ['Hn_431', 'Hn_537','Hn_441', 'Hn_554']#, 'Hn_100']#, 'Hn_100']
#samples = [20240308, 20240503, '20240319_1',  '20240319_2']
#samples = [20230404, 20230425, 20230426, 20240423, 20240812]
print(samples)

print(all_data)

mus = []
sigms = []
weighs = []

colors = ['gray','#199DC5', '#A72428', 'red', 'blue' ]
                #  'goldenrod','coral','royalblue']
fig, axes = plt.subplots(ncols=n_samples,nrows=1, figsize=(n_samples*3,2.8), 
                                 sharey=False, sharex=False, dpi=100)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12

fitcurves = []

ax = axes.ravel()
# for each sample label perform gaussian mixture with bootstrap
for ii, sample in enumerate(samples):
    
    
    if sample == 'Hn_14':
        n_comps = 1
    else:
        n_comps = 2
    # select data by label
    data = all_data[all_data['Label']==sample]
    print(data)
    
    # take the log of the Diffusion coefficients
    dapps = np.log10(data["D"].to_numpy().reshape(-1,1))
    binwidth = 0.1
    binBoundaries = np.arange(min(dapps),max(dapps), binwidth)
    #track_lengths = data['Track_len'].to_numpy().reshape(-1,1).ravel()
   # dapps = data["D"].to_numpy().reshape(-1,1)
    
    all_means = []
    all_weights = []
    all_sigmas = []
    
    dapp_id_range = range(len(dapps.ravel()))
    
    dapp_histogram = []
    for mm in range(bootstraps):
        # randomly sample data
        
        bootstrapped_indices =  np.random.choice(dapp_id_range, replace=True, 
                                    size=int(len(dapp_id_range)*boots_percent))
        
        boostrapped_d_apps =  np.asarray([dapps[index] for index in bootstrapped_indices]).reshape(-1,1)
        #boostrapped_track_lens = np.asarray([track_lengths[index] for index in bootstrapped_indices]).reshape(-1,1)
        #d_app_weights = (boostrapped_track_lens / boostrapped_track_lens.min()).astype(int)
        #weighted_dapps = np.repeat(boostrapped_d_apps, d_app_weights)
        
        
        histogram, bins = np.histogram(boostrapped_d_apps, bins=binBoundaries)
        normalized_histogram = histogram / np.sum(histogram)
        dapp_histogram.append(normalized_histogram)
        
        
        # perform gaussian mixture fitting
        if fit == 'Mix':
            
            gm = GaussianMixture(n_components=n_comps, 
                                 covariance_type='full').fit(boostrapped_d_apps)
            
            M_best = gm
            weights = gm.weights_.flatten()
            sigmas = [np.power(10,x) for x in gm.covariances_.flatten()]
            means = [np.power(10,x) for x in gm.means_.flatten()]
            
            if mm == bootstraps:
                mus.append(means)
                sigms.append(sigmas)
                weighs.append(weights)
                
    
        sort_means =  np.asarray(means).argsort()
        means.sort()
        weights = weights[sort_means]
        all_means.append(means)
        all_weights.append(weights)
        all_sigmas.append(sigmas)
        
    x = np.linspace(np.log10(10**-4), np.log10(10**1), 10000)#, min(dapps), max(dapps), 1000000)
    logprob = M_best.score_samples(x.reshape(-1,1))
    responsibilities = M_best.predict_proba(x.reshape(-1,1))   
    
    
    pdf = np.exp(logprob)
    #print(np.shape( pdf[:, np.newaxis]))
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    fitcurves.append((dapps, x, pdf, pdf_individual))
    #print(np.shape( pdf_individual))
    all_means = np.asarray(all_means).reshape(1,bootstraps,n_comps) 
    all_weights = np.asarray(all_weights).reshape(1,bootstraps,n_comps)
    #print(all_means, all_sigmas, all_weights)
    
    mean_D = []
    stdev_D = []
    mean_pi = []
    stdev_pi = []
    # results
   
    for oo in range(n_comps):
        
        mean_D.append(np.mean(all_means[:,:,oo]))
        stdev_D.append(np.std(all_means[:,:,oo]))
        mean_pi.append(np.mean(all_weights[:,:,oo]))
        stdev_pi.append(np.std(all_weights[:,:,oo]))
        
    print(sample, "N = {}".format(len(dapps)))
    print(mean_D, stdev_D, mean_pi, stdev_pi)
    
   
    xxx, bins, p = ax[ii].hist(dapps, density = True, bins = binBoundaries,
                              # weights= track_lengths,
                               histtype = 'bar', color = 'silver', 
                               alpha=1, linewidth=0, edgecolor='white')
    

    dapp_histogram_avg =np.mean(np.asarray(dapp_histogram), axis=0)
    
    dapp_histogram_CIlow = np.percentile(np.asarray(dapp_histogram), 5, axis=0)
    dapp_histogram_CIhigh = np.percentile(np.asarray(dapp_histogram), 95, axis=0)
    print(len(bins[:-1]), len(dapp_histogram_avg))
    
    d_lower_bound = np.log10(.042)
    
    ax[-1].axvline(x=d_lower_bound, linestyle='dashed', color='gray', zorder=0)
    ax[-1].fill_between(bins[:-1], dapp_histogram_CIlow, dapp_histogram_CIhigh, 
                       color=colors[ii], alpha=0.5, zorder=-1)
    ax[-1].plot(bins[:-1], dapp_histogram_avg, ls='-',
             linewidth=1, markersize=3, color=colors[ii], zorder=-1)
    
    ax[-1].set_ylim(0,.125)
    ax[-1].set_xlim(np.log10(2.5*10**-4), np.log10(10**1))
    ax[-1].set_xticks([-3, -2, -1, 0, 1 ])
    
    
    ax[-1].set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   
    ax[-1].set_ylabel('Probability density', fontsize=font_size)
    ax[-1].tick_params(axis='x',labelsize=font_size)
    ax[-1].tick_params(axis='y',labelsize=font_size)
   
    
    
    
    line = ax[ii].plot(x, pdf, '-', color=colors[ii], linewidth=1)
    indi_line = ax[ii].plot(x, pdf_individual, '--', color='darkgray', linewidth=1)
    #indi_line = ax[ii].plot(x, pdf_individual[1], '--', color='limegreen', linewidth=2)
    print(len(pdf_individual))
   
    ax[ii].set_ylim(0,1.25)
    ax[ii].set_xlim(np.log10(2.5*10**-4), np.log10(10**1))
    ax[ii].set_xticks([-3, -2, -1, 0, 1 ])
    
    ax[ii].axvline(x=d_lower_bound, linestyle='dashed', color='gray')
    ax[ii].set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   
    ax[ii].set_ylabel('Probability density', fontsize=font_size)
    ax[ii].tick_params(axis='x',labelsize=font_size)
    ax[ii].tick_params(axis='y',labelsize=font_size)

   

           
fig.tight_layout()
plt.savefig(directory[:-5] + 'Hn14_dapp.svg', dpi=300) 
plt.show()

        

###### plot all curves in one plot

hues = ['dimgray','seagreen','darkviolet', 'blue']
fig, axes = plt.subplots(ncols=1 ,nrows=1, figsize=(3,3), dpi=100)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
font_size = 12
for xx, fit in enumerate(fitcurves):
    #fitcurves.append((dapps, x, pdf, pdf_individual))
    
    xxx, bins, p = axes.hist(fit[0], density = True, bins = n_bins,
                              # weights= track_lengths,
                               histtype = 'bar', color = 'silver', 
                               alpha=1, linewidth=0, edgecolor='white')
    
    
    
    line = axes.plot(fit[1], fit[2], '-', color=hues[xx], linewidth=1)
    #indi_line = ax[ii].plot(fit[1], fit[3], '--', color=colors[ii], linewidth=2)
    #indi_line = ax[ii].plot(x, pdf_individual[1], '--', color='limegreen', linewidth=2)
   
    axes.set_ylim(0,1.1)
    axes.set_xlim(np.log10(10**-4), np.log10(10**2))
    axes.set_xticks([-4, -3, -2, -1, 0, 1, 2])
    
    
    axes.set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   
    axes.set_ylabel('Probability density', fontsize=font_size)
    axes.tick_params(axis='x',labelsize=font_size)
    axes.tick_params(axis='y',labelsize=font_size)
    
    
fig.tight_layout()
plt.savefig(directory[:-5] + 'diffusioncomp.svg', dpi=300)
plt.savefig(directory[:-5] + 'diffusioncomp.png', dpi=300) 
plt.show()    


  

'''
fig, axes = plt.subplots(ncols=1,nrows=3, figsize=(6, 11), 
                                 sharey=True, sharex=False)             
ax = axes.ravel()                
for ll,samp in enumerate(fitcurves): 
    
    dapps = samp[0]
   
    x = samp[1]
    pdf = samp[2]
    pdf_individual = samp[3]
    
    
    xxx, bins, p = ax[ll].hist(dapps, density = True, bins = n_bins,
                       histtype = 'bar', color = colors[ll],
                       linewidth=1, edgecolor='white')

    
    
    #line = ax[ll].plot(x, pdf, '-k', linewidth=3)
    indi_line = ax[ll].plot(x, pdf_individual, '-k', linewidth=3)
   # factor = max(pdf) / max(fitcurves[0][2])
   # wt_dist = fitcurves[0][3] * factor
    if ll == 2 or ll == 1:
        wt_line = ax[ll].plot(fitcurves[0][1], fitcurves[0][3], linestyle='--',color = 'navy',
                          alpha=1, linewidth=3)
   
    ax[ll].set_ylim(0,1)
    #ax[ll].set_xlim(-4,1)
    
    ax[ll].set_xlabel('log $D_{app}$ (\u03bcm\u00b2/s)', fontsize=font_size)
   # if ll == 0 or ll == 2:
    ax[ll].set_ylabel('Density', fontsize=font_size)
    ax[ll].tick_params(axis='x',labelsize=font_size)
    ax[ll].tick_params(axis='y',labelsize=font_size)
fig.tight_layout()    
plt.savefig(directory[:-5] + '20220815_Fig_Dapp_all.png', dpi=300)     
plt.show()
        
'''
    
   