#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:17:02 2022

@author: jnmc
"""

import scipy.io as sio
import numpy as np
import sys
sys.path.insert(0,'/Users/jnmc/Documents/Tesis/implenentaciones pyton/A_V_clase')
import CEBEAE_V1 as CE
sys.path.insert(0,'/Users/jnmc/Documents/Tesis/implenentaciones pyton/function_versions')
from EBEAE import ebeae
# #import EBEAEc as E
import time


nsamples = 120  # Size of the Squared Image nsamples x nsamples
#SNR = 30  # Level in dB of Gaussian Noise     SNR  = 45,50,55,60
#PSNR = 20  # Level in dB of Poisson/Shot Noise PSNR = 15,20,25,30

# Create synthetic VNIR database
# Y, Po, Ao = vnirsynth(n, nsamples, SNR, PSNR)  # Synthetic VNIR
data = sio.loadmat('/Users/jnmc/Documents/Tesis/implenentaciones pyton/VNIR_45_20.mat')
MTR = sio.loadmat('/Users/jnmc/Documents/Tesis/implenentaciones pyton/resultsmatlab.mat')
m=3
Yo=data['Y']
Po=data['Po']
Ao=data['Ao']

Am=MTR['Amatlab']
Dm=MTR['Dmatlab']
Pm=MTR['Pmatlab']
#datosLineal=E.EBEAE(Yo,n=4,initcond=4)
tfuncion = np.zeros((30,))
tclase = np.zeros((30,))

initcond = 1
rho = 1
Lambda = 0.0
epsilon = 1e-3
maxiter = 20
parallel = 0
downsampling = 0
normalization = 1
display = 0

for i in range (30):
    start=time.perf_counter()
    datos=CE.EBEAE(Yo=Yo,n=3,initcond=1, epsilon=1e-3, maxiter=20, downsampling=0, parallel=0, display=0, oae=0)
    t,results_datsos= datos.evaluate()
    end1=time.perf_counter()

    t,results=ebeae(Yo,3,[],[],0)
    end2=time.perf_counter()
    
    Pf,Af,Df,Sf,Yhf,_ = results
    tfuncion[i] = end1-start
    tclase[i] = end2-end1
    print(f'\n\ntiempo de computo EBEAE función {tfuncion[i]} s\n\n')
    print(f'\n\ntiempo de computo EBEAE clase {tclase[i]} s\n\n')
    #print(f'\n\ndiferencia entre algoritmos {end2[i]-2*end1[i]+start} s\n\n')