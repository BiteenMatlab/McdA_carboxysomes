# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 16:22:46 2025

@author: azaldegc
"""

import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lmfit import Model
import math
from scipy import stats
from sklearn.linear_model import LinearRegression


# function to look at multiple files
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames

def gaussian(x, amp1, cen1, sigma1):
        return amp1 * np.exp(-(x - cen1)**2 / (2 * sigma1**2))
    

def calc_cell_volume(area, perimeter, length, width):
    
    # coefficients for quadratic formular where we are solving for r, the radius
    # of the spherical caps of the cell. Can also be approximated to be
    # ~ 0.5 * cell width. 
    a = math.pi
    b =  -1*perimeter
    c = area
    
    # solve quadratic
    sols = np.roots([a,b,c])
    sols.sort()
    # assign r to correct solution, confirmed since h+2r is ~ cell length
    r = sols[0] 
    h = (perimeter - 2*math.pi*r)/2
    
    # return volume in um^3
    return (4/3)*(math.pi*r**2) + (2*math.pi*r*h)

# load directory with files
directory = sys.argv[1]
files = filepull(directory)
files = [file for file in files if 'intensities.csv' in file]

dfs = [pd.read_csv(file) for file in files[:]]

data = pd.concat(dfs)

name = 'Hn_532'
pixsize = 0.049 # in microns
liters_per_cubic_um = 1*10**15
avogrado = 6.02*10**23
background_signal = 1328
singmol_intensity = 11245


data_save = []

# loop through each row
for i in range(len(data)):

    # assign the area variable
    cell_area_pix = data.iloc[i]['CELL_AREA_PIX']
    
    # calculate background amount for the cell
    total_background = cell_area_pix * background_signal
    
    # subtract background amount 
    cell_intensity = data.iloc[i]['SUMMED_INTENSITY']
    cell_intensity_corrected = cell_intensity - total_background
    
    # calculate copy number
    protein_copy = cell_intensity_corrected / singmol_intensity
    
    # calculate cell volume
    cell_length = data.iloc[i]['CELL_LENGTH'] 
    cell_width = data.iloc[i]['CELL_WIDTH'] 
    cell_area = data.iloc[i]['CELL_AREA'] 
    cell_peri = data.iloc[i]['CELL_PERIMETER']*pixsize
    
    cell_vol = calc_cell_volume(cell_area, cell_peri, cell_length, cell_width)

    # calculate cellular concentration 
    molarity = (protein_copy / avogrado) / (cell_vol / liters_per_cubic_um)
    micromolar = molarity * 10**(6) 
    
    # store data
    data_save.append((data.iloc[i]['FILE'] , data.iloc[i]['SAMPLE'] , 
                      data.iloc[i]['LABEL'] , data.iloc[i]['CELL_ID'], 
                      cell_length, cell_width, cell_peri,
                      cell_area, data.iloc[i]['CELL_AREA_PIX'], 
                      data.iloc[i]['SUMMED_INTENSITY'], 
                      data.iloc[i]['AVERAGE_INTENSITY'], 
                      cell_vol, protein_copy, micromolar
                      ))
    
   
# convert data structure to dataframe

data_df = pd.DataFrame(data_save)
data_df.columns = ['FILE', 'SAMPLE', 'LABEL', 'CELL_ID', 'CELL_LENGTH',
               'CELL_WIDTH', 'CELL_PERIMETER','CELL_AREA', 'CELL_AREA_PIX',
               'SUMMED_INTENSITY', 'AVERAGE_INTENSITY',
               'CELL_VOLUME', 'PROTEIN_COPY', 'CONCENTRATION_uM']
print(data_df)
data_df.to_csv(directory[:-5] + name + '_singlecell_copynum.csv')
    
# plot


#print(data)

copy = data_df['PROTEIN_COPY'].to_numpy()
print("number of cells", len(copy))

binwidth = 50
binBoundaries = np.arange(min(copy),
                          max(copy), binwidth)

params_1 = (1,300, 50)
hist, bin_edges = np.histogram(copy, bins=binBoundaries, 
                               weights = np.zeros_like(copy) +
                               1 / copy.size)# density=True)  
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
model = Model(gaussian)
params = model.make_params(amp1=params_1[0], cen1=params_1[1], sigma1=params_1[2])

result = model.fit(hist, x=bin_centers, params=params)

fig, axes = plt.subplots(figsize=(2.9,2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
x, bins, p = plt.hist(copy,  color='gray',
                     edgecolor='k', bins=binBoundaries,
                     weights = np.zeros_like(copy) + 1 / copy.size)

print(result.params['cen1'].value, result.params['sigma1'].value)
x_smooth = np.linspace(min(copy),
                         max(copy), 1000)
yfit = gaussian(x_smooth, result.params['amp1'].value ,result.params['cen1'].value, result.params['sigma1'].value)
plt.plot(x_smooth, yfit, 'k--', label='Gaussian Fit')
plt.xlabel('McdA-mNG copy number per cell')
plt.ylabel('Normalized Frequency')
#plt.xlim(0,35000)
#plt.ylim(0, 0.2)
#plt.xticks([0, 10000, 20000, 30000])
#plt.title('Confinement Radius Histogram')
fig.tight_layout()
#plt.savefig(directory[:-5] + name + '_uncertainty.svg', dpi=300) 
#plt.savefig(directory[:-5] + name + '_uncertainty.png', dpi=300)
plt.show()


# plot two: copy vs cell length
'''
fig, axes = plt.subplots(figsize=(2.75,2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9

cell_lengths = data_df['CELL_LENGTH'].to_numpy().reshape(-1,1)
copy_number = data_df['PROTEIN_COPY'].to_numpy().reshape(-1,1)

plt.scatter(cell_lengths, copy_number, s=8, color='gray', alpha=0.5,
            edgecolor='none')

plt.xlabel('Cell length (um)', fontsize=fontsize, ) 
plt.ylabel('# of mNG molecules per cell', fontsize=fontsize)
plt.yscale('linear')
plt.ylim(-50, 1200)
plt.xticks([0.5, 1., 1.5, 2., 2.5,])
plt.xlim(0.5, 2.8)
fig.tight_layout()
#plt.savefig(directory[:-5] + label + '_nuc_by_length.svg', dpi=300) 
#plt.savefig(directory[:-5] + label + '_nuc_by_length.png', dpi=300) 
plt.show()
'''