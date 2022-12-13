#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 22:01:07 2020

@author: yangjunjie
"""

import numpy as np
import scipy.io as scio
from noiseFault import *
import os




class simulation_S8():
    '''
        simulate 8 signals
    '''
    
    def __init__(self,T,snr=30,step = 0.1,seed = None, fnr = 20,faultPos=0):
        '''

        Parameters
        ----------
        T : int
            period number
        snr : int, 
            Signal to noise ratio. The default is 30.
        step : float, optional
            Step size of time. The default is 0.1.
        seed : int, optional
            random seed. The default is None.
        fnr : float, optional
            Fault to noise ratio. The default is 20.
        faultPos : float, optional
            The position of a fault (0<= faultPos<= 1). The default is 0.

        Returns
        -------
        None.

        '''
        self.T = T
        self.snr = snr
        self.step = step
        self.t = np.arange(0.8*T*np.pi,1.8*T*np.pi,0.1)
        self.ft_index = int(faultPos*len(self.t))
        self.fnr = fnr
        
        
        
    def load(self,div,randomSeed = None,**args):
        if randomSeed is not None:
            np.random.seed(randomSeed)
        
        t = self.t
        S1 = 1+np.sin(0.1*t)

        S2 = 2*np.cos(0.25*t)**3*np.exp(-t/len(t))
        S3 = np.log(S2**2)
        S4 = S1+S2
        S5 = S1-S2
        S6 = 2*S1+S2
        S7 = S1+S3
        S8 = np.random.randn(len(t))

        
        t = t-t[0]
        signal_nf = np.c_[S1,S2,S3,S4,S5,S6,S7,S8]
        signal = wgn_signal(signal_nf,self.snr)
        
        self.t = t
        if div != 0:
            print(div)
            F,delta = FNR_inver(signal_nf[:,div-1],self.ft_index,fnr=self.fnr,\
                            snr=self.snr,step = self.step )
        
            self.F = F
            signal[:,div-1] += F
        
        return signal
    

    

class Bearing_western():
    def __init__(self,seed=None,win=2000):
        '''
        

        Parameters
        ----------
        seed : int, optional
            random seed. The default is None.
        win : int, optional
            Sample size . The default is 2000.

        Returns
        -------
        None.

        '''
        
        
        self.name = 'Bearing_western'
        self.path = '../data/western/'
        self.seed = seed
        self.win = win
        self.sampleList = [name for name in os.listdir(self.path) if name[-3:] == 'mat']
        self.current = 0

    def load(self,file=None,start = None):
        '''
            file = 97,98,99,100 (healthy)
        '''
        win = self.win
        if file is not None:
            file = str(file)+'.mat'
        else:
            file = self.sampleList[self.current]
            self.current += 1
        self.fullPath = self.path+file
        data = scio.loadmat(self.fullPath)
    
        name = file[:-4]
        if len(name) == 2 : name = '0'+name
        signal = data['X{}_DE_time'.format(name)]
        n = len(signal)
        
        if win is None:
            start = 0
            win = n
        if start is None:
            start = np.random.randint(0,len(signal)-win,1)[0]   
        
            
            
        signal = signal[start:start+win]
        try :
            Sfe = data['X{}_FE_time'.format(name)][start:start+win]
            signal = np.c_[signal,Sfe]
        except:
            pass
    
        
        return signal
    
    



