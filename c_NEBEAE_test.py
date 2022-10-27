#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 13:48:06 2022

@author: jnmc
"""
import scipy.io as sio
import numpy as np
import sys
sys.path.insert(0,'/Users/jnmc/Documents/Tesis/implenentaciones pyton/A_V_clase')
import CNEBEAE_v1 as N
sys.path.insert(0,'/Users/jnmc/Documents/Tesis/implenentaciones pyton/function_versions')
from NEBEAE_V1 import nebeae
# #import EBEAEc as E
import matplotlib.pyplot as plt
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


for i in range (30):
    start=time.perf_counter()
    datos=N.NEBEAE(Z=Yo,n=3,initcond=1, epsilon=1e-3, maxiter=20, downsampling=0, parallel=0, display=0, oae=0)
    P, A, D, Yh, S, t_D, t_A, t_P= datos.evaluate()
    end1=time.perf_counter()

    t,results=nebeae(Yo,3)
    end2=time.perf_counter()
    
    Pf,Af,Df,Sf,Yhf = results
    tfuncion[i] = end1-start
    tclase[i] = end2-end1
    print(f'\n\ntiempo de computo NEBEAE función {tfuncion[i]} s\n\n')
    print(f'\n\ntiempo de computo NEBEAE clase {tclase[i]} s\n\n')
    #print(f'\n\ndiferencia entre algoritmos {end2[i]-2*end1[i]+start} s\n\n')



# plt.figure(1,figsize=(10, 7))
# plt.clf()
# plt.plot(Po[:,0],'r')
# plt.plot(P[:,0],'b')
# plt.plot(Pf[:,0],'g')
# plt.legend(["Po","Python Class","Python function"])
# plt.plot(Po[:,1:3],'r')
# plt.plot(P[:,1:3],'b')
# plt.plot(Pf[:,1:3],'g')
# plt.title("Endmembers Estimation")
# plt.show()

# # Plot Ground-Truths and Estimated Abundances
# plt.figure(2, figsize=(10, 10))
# plt.clf()
# for i in range(1, m+1):
#     eval(f"plt.subplot(3,{m},{i})")
#     eval(f"plt.imshow(Ao[{i - 1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0],aspect='auto')")
#     plt.title(f"Endmember #{i}", fontweight="bold", fontsize=10)
    
#     eval(f"plt.subplot(3,{m},{i+m})")
#     eval(f"plt.imshow(A[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
#     if i == 2:
#         plt.title("Python Class Estimation", fontweight="bold", fontsize=10)
#     eval(f"plt.subplot(3,{m },{i+2*m})")
#     eval(f"plt.imshow(Af[{i-1},:].reshape((nsamples,nsamples)).T,extent = [0,100,100,0], aspect='auto')")
#     if i == 2:
#         plt.title("Python function Estimation", fontweight="bold", fontsize=10)
# plt.xticks(np.arange(0, 101, 20))
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# # plt.colorbar()
# plt.show()

# plt.figure(3)
# plt.clf()
# plt.subplot(211)
# plt.hist(D,range=(-0.1,0.1))
# plt.title("Python Class D estimation")
# plt.subplot(212)
# plt.hist(Df,range=(-0.1,0.1))
# plt.title("Python Function D estimation")
# plt.show()