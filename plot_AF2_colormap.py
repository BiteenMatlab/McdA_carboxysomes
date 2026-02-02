# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:28:03 2024

@author: azaldegc
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

fig, axes = plt.subplots(figsize=(4,4), dpi=300) 
plt.rcParams['font.family'] = 'Calibri'
sns.set(font="Calibri")
plt.rcParams['svg.fonttype'] = 'none'
fontsize = 8

x,y,c = zip(*np.random.rand(30,3)*4-2)

norm=plt.Normalize(39.41,97.79)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["magenta","green"])
plt.scatter(x,y,c=c, cmap=cmap, norm=norm)
cb = plt.colorbar()
cb.outline.set_color('xkcd:black')
cb.ax.tick_params(labelsize=15)


fig.tight_layout()
#plt.savefig('AF2colormap.svg', dpi=300) 
#plt.savefig(directory[:-5] + label + '_cvt.png', dpi=300) 
plt.show()