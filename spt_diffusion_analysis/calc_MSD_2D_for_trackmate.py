# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:07:37 2024

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob
from lmfit import Model
from scipy.optimize import curve_fit
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

# Define data name or label
name = 'Hn_2'
# Define the integration time for imaging (in seconds)
t_int = .2
# Define the time delay between frames in seconds
t_delay = 0.8
# Define minimum track length (in frames)
min_frames = 25
# Define the pixel size in microns
pixel_size = 0.066
# define max time lag for MSD curve
max_tau = 20
max_tau_for_fit = 5
# plot the single track MSD?
plot_single_track_msd = True




# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def calculate_msd(x, y, max_tau):
    # Initialize MSD array
    msd = []
    weights = []
    
    # Compute MSD
    for tau in range(1, max_tau+1):
        displacements = []
        for t in range((max_tau+1) - tau):
            dx = (x[t + tau] - x[t])*pixel_size
            dy = (y[t + tau] - y[t])*pixel_size
            displacements.append(dx**2 + dy**2)
        msd.append(np.mean(displacements))
        weights.append(len(displacements))
    
    return msd, np.asarray(weights)


def anomalous_diffusion(tau, D_alpha, alpha):
    return 4 * D_alpha * tau**alpha


directory = sys.argv[1]
track_files = [file for file in filepull(directory) if 'spots' in file]
# Initialize a list to store the track lengths from all files
all_msd_out = []
d_out = []
alpha_out = []
track_ids = []

# load through each file
for file in track_files:
    
    # read csv as dataframe
    df = pd.read_csv(file, header=0)
    # Calculate the radius of confinement
    vacf_out = []
    # loop through each track
    for track_id in df["TRACK_ID"].unique():
        # load track data
        track_data = df[df["TRACK_ID"] == track_id]
        n = len(track_data)
        
        # calculate radius of confinement if track is min_frames long
        # if track is longer, then only use the first min_frames n frames
        if n >= min_frames:
            # use x, y coordinates for min_frames n frames of track
            track_x_coords = track_data['POSITION_X'].to_numpy()[:]
            track_y_coords = track_data['POSITION_Y'].to_numpy()[:]
            time = track_data['POSITION_T'].to_numpy()[:min_frames]
            #print(len(track_x_coords), len(track_y_coords))
            
            # calculate MSD vs tau curve
            time_lags = np.arange(1,max_tau+1) * (t_int+t_delay)
            msd_curve, weights = calculate_msd(track_x_coords, track_y_coords, max_tau)
            weights[weights == 0] = 1  # Avoid division by zero
            sigma = np.sqrt(1 / weights)
            
            # fit to MSD function
            popt, pcov = curve_fit(anomalous_diffusion, time_lags[:max_tau_for_fit], 
                                   msd_curve[:max_tau_for_fit], 
                                   maxfev = 10000, bounds=(0, [np.inf, 2]),
                                   sigma=sigma[:max_tau_for_fit], absolute_sigma=True)
           # popt, pcov = curve_fit(anomalous_diffusion, time_lags, msd, sigma=sigma, absolute_sigma=True)
            # Extract fit parameters
            D_alpha, alpha = popt
            # calculate fitted values
            msd_fitted = anomalous_diffusion(time_lags, *popt)
            # Residuals
            residuals = msd_curve[:max_tau_for_fit] - msd_fitted[:max_tau_for_fit]
            # Calculate R^2 and RMSE
            r2 = r2_score(msd_curve[:max_tau_for_fit], msd_fitted[:max_tau_for_fit])
           
            if r2 >= .2:
                alpha_out.append(alpha)
                d_out.append(D_alpha)
                all_msd_out.append(msd_curve)
                track_ids.append(track_id)
            #print(f"Fitted parameters: D_alpha = {D_alpha}, alpha = {alpha}")
            
            if plot_single_track_msd == True:
                # plot single track MSD curves (optional)
                fig, axes = plt.subplots(figsize=(9, 3), ncols=2, dpi=150) 
                ax = axes.ravel()
                ax[1].set_title(str(alpha) + ' ' + str(r2))
                ax[0].plot(track_x_coords,track_y_coords )
                ax[1].plot(time_lags,msd_curve, marker='o')
                ax[1].plot(time_lags, anomalous_diffusion(time_lags, *popt), '-', label=f'Fit: $4D_\\alpha \\tau^\\alpha$, $D_\\alpha={D_alpha:.3f}$, $\\alpha={alpha:.3f}$')
                ax[1].set_xlabel('Time lag (tau)')
                ax[1].set_ylabel('Mean Squared Displacement')
                fig.tight_layout()
                plt.show()
            
            
            
    # Append the track lengths to the list
  
    
 
# plot the data    
def plot_data(dcoeffs):
    
    # no. of bins for all datasets
    binwidth = 0.05
    binBoundaries = np.arange(min(dcoeffs),max(dcoeffs), binwidth)
    # initiate figure
    
    fig, axes = plt.subplots(figsize=(3, 3), dpi=300) 
    plt.rcParams.update({'font.size': 9})
    plt.rcParams['font.family'] = 'Calibri'
    sns.set(font="Calibri")
    plt.rcParams['svg.fonttype'] = 'none'
    fontsize = 9
    
    plt.title("App. Diff. Coef. (n={})".format(len(dcoeffs)), 
                    fontsize=fontsize, )
    plt.xlabel('D_app (um^2/s)', fontsize=fontsize, )
    plt.ylabel('counts', fontsize=fontsize, )
   
    data_hist, bins = np.histogram(dcoeffs, bins=binBoundaries) 
    print("N bins", len(bins))
    
    x, bins, p = plt.hist(dcoeffs, bins=bins, color='forestgreen',
                            edgecolor='k', alpha=0.75)
    #plt.xscale('log')
    #plt.yscale('linear')
    #for item in p:
    #    item.set_height(item.get_height()/sum(x))
    #plt.ylim(0,40)
    #plt.xlim(0, 60)
    fig.tight_layout()
    plt.show()
    
print("N tracks", len(d_out))    
plot_data(alpha_out)
  
time_lags = np.arange(1,max_tau+1) * (t_int+t_delay)
# plot histogram
# Plot VACF
print(len(all_msd_out))

avg_msd = np.mean(all_msd_out, axis = 0)
popt, pcov = curve_fit(anomalous_diffusion, time_lags[:max_tau_for_fit], 
                       avg_msd[:max_tau_for_fit], bounds=(0, [np.inf, 2]))
# Extract fit parameters
print(popt)

std_msd1 = np.percentile(all_msd_out, 95, axis = 0)
std_msd2 = np.percentile(all_msd_out,5, axis = 0) #/ np.sqrt(len(all_msd_out))
#avg_msd = np.mean(all_msd_out, axis = 0)
#std_msd = np.percentile(all_msd_out,95, axis = 0) #/ np.sqrt(len(all_
print(avg_msd)
print(std_msd1)
print(std_msd2)
print("Diff coeff", np.mean(d_out), np.std(d_out)/np.sqrt(len(d_out)), np.percentile(d_out, 5),np.percentile(d_out, 95) )
print("alpha", np.mean(alpha_out), np.std(alpha_out))


    
labels = [name for i in range(len(track_ids))]
datatosave = [track_ids,d_out,alpha_out, labels]
dataDF = pd.DataFrame(datatosave).transpose()
dataDF.columns = ['ID','D','alpha', 'Label']
dataDF.to_csv(directory[:-5] + name +  '_mintracklen-{}'.format(min_frames) + '_Dcoeff_alpha.csv', index = False)

msd_curves_arr = np.asarray(all_msd_out)    
msd_curves_df = pd.DataFrame(msd_curves_arr.T)
msd_curves_df.insert(0,'Time', time_lags)
msd_curves_df.to_csv(directory[:-5] + name +  '_mintracklen-{}'.format(min_frames) + '_MSD_curves.csv', index = False)
print(msd_curves_df)
    

fig, axes = plt.subplots(figsize=(2.7, 2.5), dpi=150) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 7

#for msd in all_msd_out:
    
 #   plt.plot(time_lags, np.log10(msd), color='gray',alpha=0.1,)
    
plt.plot(time_lags, avg_msd, marker='o', color='black',linestyle='None', markersize=2)
#plt.fill_between(time_lags, std_msd2, std_msd1, alpha=0.25, linewidth=0,color='gray')
plt.xlabel('Time Lag (s)')
plt.ylabel('MSD')
plt.xscale('log')
plt.xlim(0.9, 50)
plt.ylim(10**-5, 10**-1)
plt.yscale('log')


fig.tight_layout()
#plt.savefig(directory[:-5] + name + '_msd.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_msd.png', dpi=300) 
plt.show()
