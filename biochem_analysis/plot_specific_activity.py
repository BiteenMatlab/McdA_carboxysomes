# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:22:28 2024

@author: azaldegc
"""

import sys
import numpy as np
import glob
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt



# Pull files from directory
def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames
directory  = sys.argv[1]
filenames = filepull(directory)


df = pd.read_csv(filenames[0])

average = df['Specific activity'].to_numpy()
sd = df['error'].to_numpy()

fig, axes = plt.subplots(figsize=(3.58, 3), 
                        dpi=150)

plt.rcParams.update({'font.size': 9})
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 9

samples = ['WT', 'K20A', 'R158E']
hues = ['gray', 'blue', 'green']
error = []

#Establish order of x-values and hues; retreive number of hues
order = set(df["DNA"]); hue_order=set(df["Sample"]); n_hues = len(hue_order)

ax = sns.barplot(x="DNA", y="Specific activity", hue='Sample',
            data=df, palette=hues,
            linewidth=2, edgecolor="black",)

# Get the bar width of the plot
bar_width = ax.patches[0].get_width()
#Calculate offsets for number of hues provided
offset = np.linspace(-n_hues / 2, n_hues / 2, n_hues)*bar_width*n_hues/(n_hues+1); # Scale offset by number of hues, (dividing by n_hues+1 accounts for the half bar widths to the left and right of the first and last error bars.
#Create dictionary to map x values and hues to specific x-positions and offsets
x_dict = dict((x_val,x_pos) for x_pos,x_val in list(enumerate(order)))
hue_dict = dict((hue_pos,hue_val) for hue_val,hue_pos in list(zip(offset,hue_order)))
#Map the x-position and offset of each record in the dataset
x_values = np.array([x_dict[x] for x in df["DNA"]]);
hue_values = np.array([hue_dict[x] for x in df["Sample"]]);
print(x_values)
print(hue_values)
#Overlay the error bars onto plot
ax.errorbar(x = x_values+hue_values, y = average, yerr=sd, fmt='none', c= 'black', capsize = 2)

plt.ylim(0,40)
axes.get_legend().remove()
fig.tight_layout()
plt.savefig(directory[:-5] + 'specific_activity_comp.svg', dpi=300) 
plt.savefig(directory[:-5] + 'specific_activity_comp.png', dpi=300) 
plt.show()