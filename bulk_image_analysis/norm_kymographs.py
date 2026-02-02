# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:06:37 2024

@author: azaldegc
"""

# import modules
import sys
import numpy as np
import glob
import tifffile as tif
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import ListedColormap


def filepull(directory):
    """
    Finds files in directory and returns them as list

    Parameters
    ----------
    directory : string ; path to directory

    Returns
    -------
    filenames : list ; list of filename strings

    """
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]

    return filenames


directory = sys.argv[1] # directory path
files = filepull(directory) # load filenames
files = [file for file in files if "kymos" in file] # keep files with 'kymos' in name

frame_rate = 0.16 # in mins
label = 'McdA'


kymographs = []

for file in files[:]:
    
    kymo = tif.imread(file)
    norm_kymo = []
    norm_kymo = np.zeros(np.shape(kymo))
    for ii, t in enumerate(range(len(kymo))):
        
        norm_kymo[ii] = (kymo[ii] - np.amin(kymo[ii])) / (np.amax(kymo[ii]) - np.amin(kymo[ii]))
        
    kymographs.append(norm_kymo)
          
time_axis = [time*frame_rate for time in range(len(kymo))]  
    

# plot kymographs
fig,axes = plt.subplots(figsize=(3, len(kymographs)*3),
                        nrows=len(kymographs)+1,ncols=1, 
                        sharex=False, sharey=True, dpi=100)    
ax = axes.ravel()
font_size = 9
plt.rcParams.update({'font.size': font_size})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 230/256, N)
vals[:, 1] = np.linspace(0/256, 230/256, N)
vals[:, 2] = np.linspace(0, 5/256, N)
newcmp = ListedColormap(vals)


for ii in range(len(kymographs)):
    ax[ii].imshow(kymographs[ii], cmap=newcmp, vmin=0.0,vmax=1)
    print(np.shape(kymographs[ii]))
    
    ax[ii].set_xticks([]) 
    ax[ii].set_xticks([0,np.shape(kymographs[ii])[1] - 1])
    ax[ii].set_xticklabels(['-1', '1'],fontsize=font_size )
    ax[ii].xaxis.tick_top()
    
cmap = cm.ScalarMappable(cmap=newcmp)    
cbar = fig.colorbar(cmap, ax=ax[-1],)
cbar.outline.set_color('black')
cbar.ax.set_ylabel('Normalized intensity (a.u.)', fontsize=font_size)
cbar.ax.tick_params(labelsize=font_size)   
                                          
ax[0].set_yticks([0, 6, 12, 18])#, 15, 30])
ax[0].set_yticklabels(['0', '18', '36', '48'],fontsize=font_size )
ax[0].set_ylabel('Time (min)', fontsize=font_size)
fig.tight_layout()
plt.savefig(directory[:-5] + label + '_kymos.svg', dpi=300) 
plt.savefig(directory[:-5] + label + '_kymos.png', dpi=300) 
plt.show()