#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:41:09 2023

@author: elliemorgenroth
"""

# Import all necessary files
import numpy as np

from plot import *
import pandas as pd
from munkres import Munkres

DA_PLS = pd.read_pickle(r"../pkl/Discrete_Appraisal_PLS.pkl")

bmodes = ['Discrete','Appraisal']

for mode in bmodes:
    B_PLS = A_PLS = pd.read_pickle(f"../pkl/pls_res_{mode}.pkl")['U']
    if mode == 'Appraisal':
        A_PLS = DA_PLS['U']
    elif mode == 'Discrete':
        A_PLS = DA_PLS['V']
    matrix = np.corrcoef([A_PLS,B_PLS])
    #m = Munkres()
    #indexes = m.compute(matrix)
    
GM = '/Volumes/Data2/Movies_Emo/Ellie/reg/standard_mask.nii.gz'
GM = image.load_img(GM)
GM = GM.get_fdata()
GM = GM.reshape(-1)

Abrain = (r"../Nifti/LVS_Discrete_all.nii.gz")
Dbrain = (r"../Nifti/LVS_Appraisal_all.nii.gz")

Adata = image.load_img(Abrain)
Adata = Adata.get_fdata()
Adata = Adata.reshape(-1,Adata.shape[-1])

Ddata = image.load_img(Dbrain)
Ddata = Ddata.get_fdata()
Ddata = Ddata.reshape(-1,Ddata.shape[-1])

matrix = np.corrcoef(Adata,Ddata)