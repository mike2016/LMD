#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:54:35 2021

@author: yangjunjie
"""


import numpy as np
from noiseFault import *
from scipy.stats import wasserstein_distance,entropy
from sklearn.decomposition import KernelPCA,PCA
import math
import scipy.io as scio
#import pylab as pl
import scipy.signal as signal
from scipy import fftpack  
from scipy import interpolate
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f as Fdistribution
from scipy.stats import norm
from scipy.stats import chi2
import pickle
from sklearn.cluster import MeanShift
from scipy.linalg import sqrtm
from numpy import polyfit
from scipy.optimize import minimize
from saveFigure import figureSave
from sklearn.neighbors import KernelDensity
from numpy import linalg as LA
from sklearn.decomposition import FastICA



''' 
    -----------------------------------------------------
    |                                                   |
    |                      LMD                          |
    |                                                   |
    -----------------------------------------------------
'''
class LMD():
    def __init__(self,alpha=0.99,minR='auto',envelope=0,Ulimit=1,EPDweight = 0.8):
        '''
        

        Parameters
        ----------
        alpha : float, optional
            Confidence level. The default is 0.99.
        minR : float, optional
            local region radius; when set to 'auto', it will be optimized. The default is 'auto'.
        envelope : int, optional
            Sample number for create the envelope of the result. The default is 0.
        Ulimit : int, optional
            Minimum number of local samples. The default is 1.
        EPDweight : float, optional
            Weight for EPD-Cusum calculation. The default is 0.8.

        Returns
        -------
        None.

        '''
        self.name = 'LMD'
        self.params = {}
        self.params['theta'] = None
        self.params['minR'] = minR
        self.params['envelope'] = envelope
        self.params['ulimit'] = Ulimit
        self.params['alpha'] = alpha
        self.params['temp'] = {}
        self.params['EPDweight'] = EPDweight
        
    
    def getThreshold(self,D,alpha):
        hin,bar = np.histogram(D.reshape(-1),bins=500)
        bar = (bar[0:-1]+bar[1:])/2
        P = np.cumsum(hin)/len(D.reshape(-1))
        th = bar[P >=alpha][0]
        return th
    
    def emCDF(self,data):
        p = np.array([(data<i).mean() for i in self.params['temp']['X'] ])
        self.params['temp']['p'] = p

    def GEV(self,theta,x):
        mu = theta[0]
        sigma = theta[1]
        lada =theta[2]
        return  1-np.exp( -( 1+lada*((mu-x)/sigma) )**(-1/lada)  )
    
    def InvGEV(self,theta,y):
        mu = theta[0]
        sigma = theta[1]
        lada =theta[2]
        return mu+(sigma/lada)-sigma/lada*( (-np.log(1-y))**(-lada) ) 
    
    def rosen(self,theta):
        return ((self.GEV(theta,self.params['temp']['X'])-self.params['temp']['p'])**2).sum()

    
    
    def getThreshold2(self,result,alpha=0.995):
        '''
            Determine a threshold for LMD result using GEV model

        '''
        
        print('Estimating CDF...')
        X = self.params['temp']['X'] = np.arange(0,result.max(),result.max()/200)
        self.emCDF(result)
        x0 = np.array([0.33,0.14,-0.4])
        res = minimize(self.rosen, x0, method='nelder-mead',
                 options={'xatol': 1e-5, 'disp': False,'return_all':True})
        self.params['theta'] = theta = res.x
        th = self.InvGEV(theta,alpha)
    
        return th
     
    
    def dertermine_minR(self,ref,return_all=False,Ulimit=None,\
                        maxSearch=1000,minErr=1e-4):
        '''
            Local region radius optimization

        
        '''
        print('Optimizing local radius...')

      
        corr = np.cov(ref.T)
        
        d = self.distance(ref,ref).reshape(-1)
        rightR = d.max()
        d[d==0] = np.inf
        leftR = d.min()
        
        
        for i in range(maxSearch):
            R1 = (rightR-leftR)/3+leftR
            an1,w = self.anchorGen(ref,minR=R1,Ulimit=Ulimit)
            D1 = self.distance(an1,ref,corr= corr)
            loss1 = D1.min(axis=0).mean()
            
            
            R2 = (rightR-leftR)/3*2+leftR
            an2,w = self.anchorGen(ref,minR=R2,Ulimit=Ulimit)
            D2 = self.distance(an2,ref,corr= corr)
            loss2 = D2.min(axis=0).mean()
            
            if np.abs(loss2-loss1) < minErr :
                break
            elif loss1 < loss2 :
               rightR = R2
            else:
               leftR = R1
            #print('{} : R1-{}; R2-{}; L1-{}; L2-{};'.format(i,leftR,rightR,loss1,loss2))
               
             

        R_opt = (R2+R1)/2
        loss = (loss1+loss2)/2
        
        print('Optimal R: {}; Loss:{} ; search time:{}'.format(R_opt,loss, i) )
        if return_all:
            return R_opt,loss
        else:
            return R_opt
            
    
    
        
        
    def anchorGen(self,data,minR,wRule='V3',Ulimit=0):
       
        '''
            Select sample according to the Mahanobis distance and
            remove outliters form the sample according to the sample
            number in th region
            
        '''
        corr = np.cov(data.T)
        D = self.distance(data,corr=corr)
        DsortIndex = np.argsort(D)
        
        sample_num = len(data)
        D = self.distance(data,data,corr=corr)
        LsortIndex = np.arange(sample_num)
        
        anchors = []
        dropIndex = []
        weight = []
        for i in DsortIndex[:int(sample_num*1)]:
            if i in dropIndex:
                continue
            L = D[i,:]
            indexs = LsortIndex[L<=minR]   
            if len(indexs) > Ulimit:
                anchors.append(data[indexs].mean(axis=0))
                weight.append(len(indexs))
            dropIndex += list(indexs)
            D[:,indexs] = np.inf
        if len(anchors) == 0:
            anchors.append(data.mean(axis=0))
            weight.append(len(data))
        anchors = np.array(anchors)
        weight = np.array(weight)
        #print('anchors size: {}'.format(anchors.shape))
        return anchors,weight
            
        

    
    def distance(self,ref,signal=None,corr = None,sigma=None):
        '''
            Calculate Mahalanobis distance
        '''
        
        ref_num = len(ref)
        if len(ref.shape) == 1:
            ref = ref.reshape(ref_num,1)
            sigma = np.arrsy([[1]])
        if signal is None:
            signal = ref.mean(axis=0)
            signal = signal.reshape(1,-1)
        
        if len(signal.shape) == 1:
            signal = signal.reshape(-1,1)
        
        signal_num = len(signal)
        D = 0
        
        D = np.zeros((ref_num,signal_num))
        if sigma is not None:
            pass
        elif corr is not None :
            sigma = np.linalg.inv(sqrtm(corr))
        else:
            corr = np.cov(ref.T)
            sigma = np.linalg.inv(sqrtm(corr))
            
        for i in range(ref_num):
                D[i,:] = np.sqrt((((ref[i,:]-signal).dot(sigma) )**2 ).sum(axis=1) )
        

        
        if D.shape[-1] == 1:
            D = D.reshape(-1)
        return D
    
    def DFdetect(self,signal,anchors,corr=None,envelope=0,return_pre=False): 
        D = self.distance(anchors,signal,corr= corr)
        result = D.min(axis=0)
        index = D.argmin(axis=0)
        S = anchors[index,:]
            
        if envelope!=0:
            result = self.envelopeCreate(result,envelope)
        if return_pre :
            return result,S
        else:
            return result
        
        
    def saveParams(self,path):
        f = open(path, 'wb')
        pickle.dump(self.params, f)
        f.close()
        print('save parameter to: {}'.format(path))
        
    def loadParams(self,path):
        f = open(path, 'rb')
        self.params = pickle.load(f)
        f.close()
        

    def fit(self,ref):
        minR = self.params['minR'] 
        envelope = self.params['envelope'] 
        Ulimit = self.params['ulimit'] 
        alpha = self.params['alpha'] 
        
        if minR == 'auto':
            minR = self.dertermine_minR(ref,Ulimit=Ulimit)

        
        corr = np.cov(ref.T)
        anchors,w = self.anchorGen(ref,minR,Ulimit=Ulimit)   
        print('Anchor number: {}'.format(anchors.shape[0]))
        
            
        result = self.DFdetect(ref,anchors=anchors,\
                               corr = corr,envelope=envelope)
        
        th = self.getThreshold2(result,alpha)
        
        
        EPDw = self.params['EPDweight']
        y = th*np.ones(len(result))
        resultEPD = self.EPD(result,y,EPDw,alpha=1-alpha)
        
        
        self.params['anchor'] = anchors
        self.params['anWeight'] = w
        self.params['anTh'] = th
        self.params['corr'] = corr
        self.params['sVar'] = ref.std(axis=0)
        self.params['Ug'] = self.getThreshold(resultEPD,alpha)
        
        
        
    def transform(self,signal,EPDdetect = False):
        '''
            Compute LMD result
        '''
        
        envelope = self.params['envelope']
        minR = self.params['minR']
        corr = self.params['corr'] 
        anchors = self.params['anchor']
        w = self.params['anWeight']
        result = self.DFdetect(signal,anchors=anchors,\
                               corr=corr,envelope=envelope,return_pre=False)
        
        if EPDdetect:
            EPDw = self.params['EPDweight']
            alpha = self.params['alpha']
            th = self.params['anTh']
            y = th*np.ones(len(result))
            result = self.EPD(result,y,EPDw,alpha=1-alpha)
        return result
    
    def getAnchors(self,signal):
        envelope = self.params['envelope']
        minR = self.params['minR']
        corr = self.params['corr'] 
        anchors = self.params['anchor']
        w = self.params['anWeight']
        
        D = self.distance(anchors,signal,corr= corr)
        
        result = D.min(axis=0)
        index = D.argmin(axis=0)
        S = anchors[index,:]
       
        return S
        
    
    def getAnchorsIndex(self,signal):
        envelope = self.params['envelope']
        minR = self.params['minR']
        corr = self.params['corr'] 
        anchors = self.params['anchor']
        w = self.params['anWeight']
        
        D = self.distance(anchors,signal,corr= corr)
       
        result = D.min(axis=0)
        index = D.argmin(axis=0)
        return index
       
        
    
    
    def EPD(self,x,y,w,alpha=0.001):
        I = ( x>y )*1
        p = np.zeros(I.shape)
        r = np.zeros(I.shape)
        R = np.zeros(I.shape)
        G = np.zeros(I.shape)
        min1 = 0
        for i in range(1,len(I)):
            W = max([1/i,w])
            p[i] = max([(1-W)*p[i-1] + W*I[i] , 10e-5])
            G[i] = max([G[i-1]+np.log(p[i]/alpha),0])
          
        return G
    
    
    
    def fIsolateEstimate(self,signal,tStep):
        '''
        
            Faulty variable isolation and fault increasing rate estimation

        '''
        sVar = self.params['sVar']
        S = self.getAnchors(signal)
        G = (signal-S)/sVar
        Fc = np.argmax(np.abs(G.mean(axis=0)))
        
        result = self.transform(signal,EPDdetect = False)
        corr = self.params['corr'] 
        result = result**2
        corr_12 = sqrtm(np.linalg.inv(corr))
        sigma = (corr_12[Fc,:]**2).sum()
        est = 0
        y = result
        t = np.arange(len(y))*tStep
        coeff = polyfit(t, y, 2)
        est =  np.sqrt(np.abs(coeff[0])/sigma)
        return Fc+1,est
    


    
    
    
    