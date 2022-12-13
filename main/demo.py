#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 19:28:04 2022

@author: yangjunjie
"""


import _init_paths
from datasets.data import simulation_S8
from algorithm.LMD import LMD
import matplotlib.pyplot as plt
import numpy as np


# load data 
faultVariable = 2
step = 0.1
data = simulation_S8(T=30,snr=20,step = step,fnr = 40,faultPos=0.2)
ref = data.load(0)
t = data.t
faultOccTime = t[data.ft_index]


# initiate and train LMD model 
lmdM = LMD(alpha=0.99,minR=1.68,Ulimit=2,EPDweight = 0.5)
lmdM.fit(ref)   


# genrate faulty signals
signal = data.load(faultVariable)

# Compute LMD and EPD-Cusum results
Dl = lmdM.transform(signal,EPDdetect=False) 
Depd = lmdM.transform(signal,EPDdetect=True) 

threshold = lmdM.params['anTh']
controlLimit = lmdM.params['Ug']

# 
detectTimeIndex = np.argmax(controlLimit< Depd)
detectTime = t[detectTimeIndex]

# Faulty variable isolation and fault increasing rate estimation
Fc,delta = lmdM.fIsolateEstimate(signal[detectTimeIndex:,:],tStep=step)

print('''The dignosis results are given below: 
      The fault detected time is {};
      The faulty variable is {};
      The increasing rate is {}.'''.format(detectTime,Fc,delta))

fig = plt.figure()
layout = (2,1)
ax0 = plt.subplot2grid(layout, (0,0))
ax0.plot(Dl,label='LMD result',color='c')
ax0.plot(threshold*np.ones(len(Dl)),label='threshold',color='r')
ax0.legend(loc="upper left")

ax1 = plt.subplot2grid(layout, (1, 0))
ax1.plot(Depd,label='EPD Cusum result',color='c')
ax1.plot(controlLimit*np.ones(len(Dl)),label='control limit',color='r')
ax1.legend(loc="upper left")