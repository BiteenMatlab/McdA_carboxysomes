# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 08:21:55 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from scipy import stats
from sklearn.linear_model import LinearRegression

def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames


def linear(x, m, b):
    return m*x  + b

label = '2_vs_100'




directory  = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'nucleoid_analysis' in file]
samples = []
dfs = []
for file in filenames[:]:
    df = pd.read_csv(file,index_col=0)
    dfs.append(df)
all_data = pd.concat(dfs).reset_index()
print(all_data)

fig, axes = plt.subplots(figsize=(3, 3), dpi=300) 
plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9
samples = ['Hn_2']
colors = ['gray']
print(samples)
for ii, sample in enumerate(samples):
    print(sample)
    sample_data = all_data[all_data['Sample']==sample]
    
    
    cell_lengths = sample_data['Cell_Length'].to_numpy().reshape(-1,1)
    print("avg cell length: ", np.mean(cell_lengths))
    nuc_lengths = sample_data['Nucleoid_Length'].to_numpy().reshape(-1,1)
    print(len(cell_lengths), len(nuc_lengths))
    slope, intercept, r_value, p_value, se = stats.linregress(cell_lengths[:,0], nuc_lengths[:,0])
    regr = LinearRegression(fit_intercept=False)

    # Train the model using the training sets
    regr.fit(cell_lengths, nuc_lengths)
    x = np.asarray([0, 0.5, 1, 1.5, 2, 2.5, 3]).reshape(-1,1)
    # Make predictions using the testing set
    y_pred = regr.predict(x)

    # The coefficients
    print("Coefficients: \n", regr.coef_)
    # The mean squared error
    #print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
    ## The coefficient of determination: 1 is perfect prediction
    #print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

    # Plot outputs
    #plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    #plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    # Print the slope, intercept, and R-squared value
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value**2}")
    
    
    #line = slope * x + intercept
    plt.plot(x, y_pred, color='black',linestyle='dashed', alpha =1)

    plt.scatter(cell_lengths, nuc_lengths, s=8, color=colors[ii], alpha=0.5,
                edgecolor='none')


plt.xlabel('Cell length (um)', fontsize=fontsize, ) 
plt.ylabel('Nucleoid length (um)', fontsize=fontsize)
plt.yscale('linear')
plt.ylim(0, 2.8)
plt.xticks([0.5, 1., 1.5, 2., 2.5,])
plt.xlim(0.5, 2.8)
fig.tight_layout()
plt.savefig(directory[:-5] + label + '_nuc_by_length.svg', dpi=300) 
plt.savefig(directory[:-5] + label + '_nuc_by_length.png', dpi=300) 
plt.show()