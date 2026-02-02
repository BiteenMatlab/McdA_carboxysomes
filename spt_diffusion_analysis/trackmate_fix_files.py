# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:10:50 2024

@author: azaldegc
"""

import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np
import pandas as pd
import sys
import scipy.stats as ss
import json


def filepull(directory):
    '''Finds images files in directory and returns them'''
    # create list of image files in directory
    filenames = [img for img in glob.glob(directory)]   
    
    return filenames




directory  = sys.argv[1]
filenames = filepull(directory)
filenames = [file for file in filenames if 'spots' in file]

for file in filenames[1:]:
    
    df = pd.read_csv(file ,index_col=None)
    print(df)
    newdf = df.drop([0, 1, 2])
    
    print(newdf)
    newdf.to_csv(file)
    