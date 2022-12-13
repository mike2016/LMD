#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:17:13 2021

@author: yangjunjie
"""


import numpy as np


def wgn_signal(x,snr,seed = None):
    
    if seed is not None:
        np.random.seed(seed)
    
    
    Ps = (x**2).mean(axis=0) 
    r = 10**(snr/10)
    sigma = np.sqrt(Ps / r)
    
    if len(x.shape) ==2:
        noise = np.random.randn(x.shape[0],x.shape[1])
    else:
        noise = np.random.randn(len(x))
        
    noise = (noise- noise.mean(axis=0))/noise.std(axis=0)
        
    v = noise * sigma
    y = x+v
    return y


def FNR(x,F,snr):
    Ps = (x**2).mean()
    sigma2 = Ps/ 10**(snr/10)
    Pf = (F**2).mean()
    return 10*np.log10(Pf/sigma2)

def SFR(x,F):
    Ps = (x**2).mean()
    Pf = (F**2).mean()
    return 10*np.log10(Ps/Pf)


def FNR_inver(x,ft_index,fnr,snr,step=0.1,Type='incipient'):
    
    N = len(x)
    Ps = (x**2).mean()
    
    Pf = Ps*(10**((fnr-snr)/10))
    
    if Type == 'incipient':

        delta = np.sqrt( Ps*(10**((fnr-snr)/10)) *6/ (N-ft_index+2)/ (2*N-2*ft_index+3)  )
        t = np.arange(N)*step
        F = np.r_[np.zeros(ft_index-1),(t[ft_index-1:]-t[ft_index-2])*delta]
    else:
        delta = np.sqrt(Pf)
        
        F = np.r_[np.zeros(ft_index-1),np.ones(N-ft_index+1)*delta]
    
    return F,delta

def FNR_SNR_inver(x,ft_index,fnr,sfr,step=0.1,Type='incipient'):
    N = len(x)
    Ps = (x**2).mean()
    snr = sfr+fnr
    sigma2 = Ps/ (10**(snr/10))
    Pf = Ps*(10**((fnr-snr)/10))
    
    if Type == 'incipient':
        delta = np.sqrt( Ps*(10**((fnr-snr)/10)) *6/ (N-ft_index+2)/ (2*N-2*ft_index+3)  )
        t = np.arange(N)*step
        F = np.r_[np.zeros(ft_index-1),(t[ft_index-1:]-t[ft_index-2])*delta]
    else:
        delta = np.sqrt(Pf)
        
        F = np.r_[np.zeros(ft_index-1),np.ones(N-ft_index+1)*delta]
    
    return F,delta,snr


def SFR_inver(x,ft_index,sfr,step=0.1,Type='incipient'):
    N = len(x)
    Ps = (x**2).mean() 
    Pf = Ps/(10**(sfr/10) ) 
    if Type == 'incipient':
        delta = np.sqrt(Pf*6*N/(N-ft_index+1)/(N-ft_index+2)/\
                        (2*N-2*ft_index+3)/step**2)
        
        t = np.arange(N)*step
        F = np.r_[np.zeros(ft_index-1),(t[ft_index-1:]-t[ft_index-2])*delta]
    else:
        delta = np.sqrt(Pf*N/(N-ft_index+1))
        F = np.r_[np.zeros(ft_index-1),np.ones(N-ft_index+1)*delta]
    
    return F,delta

def outliers(sig,p,gain):
    if p <= 0:
        return sig
    N = len(sig)
    sigma0 = sig.std(axis=0)
    out = np.random.randn(sig.shape[0],sig.shape[1])
    out = (out - out.mean(axis=0))/out.std(axis=0)*sigma0*gain
    index = np.arange(N)
    np.random.shuffle(index)
    num = int(p*N)
    sig2 = sig.copy()
    sig2[index[:num],:] += out[index[:num],:]
    return sig2


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sig = np.random.randn(1200)
    F,delta = FNR_inver(sig,1000,30,30,step=1,Type='incipient')
    fnr = FNR(sig,F[999:],30)
    
    print(fnr)
    
    
    
    