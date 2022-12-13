#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 18:30:54 2021

@author: yangjunjie
"""



import scipy.io as scio
import os
import numpy as np


def figureSave(name,data1,data2=None,label = 'curve',line = 'b',xlabel='x',\
               ylabel='y',path='./'):
    '''
        Save figure's data for matlab
    '''
    if data2 is None:
        x = np.arange(len(data1)).reshape(1,-1)
        y = data1.reshape(1,-1)
    else:
        x = data1.reshape(1,-1)
        y = data2.reshape(1,-1)
        
    fileName = path+name+'.mat'
    if os.path.exists(fileName):
        data = scio.loadmat(fileName)
        data['x'] = np.r_[data['x'],x]
        data['y'] = np.r_[data['y'],y]
        data['labels'] = list(data['labels'])+[label]
        data['line'] = list(data['line'])+[line]
    else:
        data = {}
        data['x'] = x
        data['y'] = y
        data['labels'] = [label]
        data['line'] = [line]
        
    data['Xaxis'] = xlabel
    data['Yaxis'] = ylabel
    scio.savemat(fileName, data)
    
    
    
if __name__ == '__main__':
    name = '1111'
    data1 = np.arange(100)
    data2 = np.random.randn(100)
    figureSave(name,data1,data2,label = 'b')
    figureSave(name,data1,data2,label = 'r')
    figureSave(name,data1,data2,label = 'rd')
    data = scio.loadmat('../output/figure/1111.mat')
    print(data['labels'])
    print(data['line'])