# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:23:41 2025

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


directory = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'copynum' in file]

dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
    #samples.append(df['Label'].iloc[0])
data_df = pd.concat(dfs).reset_index()

samples = ['Hn_2', 'Hn_532']
colors = ['gray','#199DC5']



fig, axes = plt.subplots(figsize=(2.9,2.5), dpi=300) 
plt.rcParams.update({'font.size': 9})    
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9

for ii, sample in enumerate(samples):
    
    
    if sample == 'Hn_14':
        n_comps = 1
    else:
        n_comps = 2
    # select data by label
    data = data_df[data_df['SAMPLE']==sample]
    
    # plot two: copy vs cell length
    cell_lengths = data['CELL_LENGTH'].to_numpy().reshape(-1,1)
    print(len(cell_lengths))
    copy_number = data['PROTEIN_COPY'].to_numpy().reshape(-1,1)
    
    plt.scatter(cell_lengths, copy_number, s=8, color=colors[ii], alpha=0.75,
            edgecolor='none')
    
    slope, intercept, r_value, p_value, se = stats.linregress(cell_lengths[:,0], copy_number[:,0])
    regr = LinearRegression(fit_intercept=False)

    # Train the model using the training sets
    regr.fit(cell_lengths, copy_number)
    x = np.asarray([0, 0.5, 1, 1.5, 2, 2.5, 3]).reshape(-1,1)
    # Make predictions using the testing set
    y_pred = regr.predict(x)
    #line = slope * x + intercept
    plt.plot(x, y_pred, color='black',linestyle='dashed', alpha =1)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value**2}")

plt.xlabel('Cell length (um)', fontsize=fontsize, ) 
plt.ylabel('# of mNG molecules per cell', fontsize=fontsize)
plt.yscale('linear')
plt.ylim(-75, 1200)
plt.xticks([0.5, 1., 1.5, 2., 2.5,])
plt.xlim(0.5, 2.8)
fig.tight_layout()
#plt.savefig(directory[:-5] + label + '_nuc_by_length.svg', dpi=300) 
#plt.savefig(directory[:-5] + label + '_nuc_by_length.png', dpi=300) 
plt.show()