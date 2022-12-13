#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 14:24:55 2020

@author: yangjunjie
"""

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score,recall_score,f1_score



font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,}

def resample(fpr,tpr,step):
    fpr2 = np.arange(0,1+1e-5,step)
    tpr2 = np.zeros(fpr2.shape)
    tpr2[0] = tpr[0]
    tpr2[-1] = tpr[-1]
    currentIndex = 0
    for i,x in enumerate(fpr2[1:-1]):
        # print(x)
        # print(fpr[currentIndex+1])
        while x >= fpr[currentIndex+1] :
            currentIndex += 1
        k = (tpr[currentIndex+1]-tpr[currentIndex])/(fpr[currentIndex+1]-fpr[currentIndex])
        tpr2[i+1] = k*(x-fpr[currentIndex])+tpr[currentIndex]
    return fpr2,tpr2
            
def auc_point(label,score,resampleStep = None):
    if resampleStep is  None:
        fpr,tpr,threshold = roc_curve(label,score)
        return fpr,tpr
    if len(score.shape) == 1:
        label = label.reshape(1,-1)
        score = score.reshape(1,-1)
    
    fpr = np.arange(0,1+1e-5,resampleStep)
    fpr[0] = 0
    fpr[-1] = 1
    varNum = len(label)
    pointNum = len(fpr)
    fprMat = np.tile(fpr,[varNum,1])
    tprMat = np.zeros((varNum,pointNum))
    for i in range(varNum): 
        fpr,tpr,threshold = roc_curve(label[i,:],score[i,:])
        fpr,tpr = resample(fpr,tpr,resampleStep)
        tprMat[i,:] = tpr
    return fprMat,tprMat
        
        

def ROC_AUC(label,score,Plot=False,lw = 3,figsize=(10, 10),legend=None,\
            optTh =False):
    try:
        r,c = label.shape
    except:
        c = len(label)
        r = 1
        label = label.reshape(1,c)
        score = score.reshape(1,c)
        
    aucValue = np.zeros(r)
    thresholds = np.zeros(r)
    Flag = True
    colors = ['b','g','c','r','m','y','k','rosybrown','chocolate','orange','olivedrab'\
              'teal','steelblue','indigo','deeppink']*2
    if legend is None:
        legend = ['ROC curve']*r

    for i in range(r):
        fpr,tpr,threshold = roc_curve(label[i,:],score[i,:])
        aucValue[i] = auc(fpr,tpr)
        index = np.argmax(tpr-fpr)
        thresholds[i] = threshold[index]
        if Plot:
            if Flag:
                #plt.figure()
                plt.figure(figsize=figsize,dpi=200)
                Flag = False
            
            plt.plot(fpr, tpr, color=colors[i],\
                     lw=lw, label='${}$(auc = {:.3f})'.format(legend[i],aucValue[i]))
            plt.legend(loc="lower right",prop=font2)
            
    if Plot:
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.tick_params(labelsize=24)
        plt.xlabel('False Positive Rate',font)
        plt.ylabel('True Positive Rate',font)
        plt.title('Receiver operating characteristic',font)
    if optTh:
        return aucValue,thresholds
    return aucValue
        


def accuracy(label,result,th = None):
    if th is not None:
        ft_index = np.argmax(label)
        acc = (np.sum(result[:ft_index] < th )+np.sum(result[ft_index:] >= th ))/len(result)
    else:
        acc = np.sum(label==result)/len(result)
    return acc

def F1(label,result):
    labelName = list(set(label))
    TP = np.mean( result[ label==labelName[1] ] >0 )
    FP = np.mean( result[ label==labelName[0] ] >0 )
    FN = np.mean( result[ label==labelName[0] ] <=0 )
    pre = TP/(TP+FP)
    recall = TP/(TP+FN)
    f = 2*(pre*recall)/(pre+recall)
    return f

def genThreshold(data,num=10):
    data = np.sort(data)
    threshold = np.array([data[i] for i in range(0,len(data),num)])
    threshold = np.r_[threshold,np.max(data)+1]
    threshold = np.array(list(set(threshold)))
    return np.sort(threshold)[::-1]

def roc_curve2(label,result,stepNum = 10):
    labelName = np.unique(label)
    threshold = genThreshold(result,num=stepNum)
    nPart = result[label == labelName[0]]
    pPart = result[label == labelName[1]]
    N = len(result)
    fpr = np.array([(nPart>=t).mean() for t in threshold])
    tpr = np.array([(pPart>=t).mean() for t in threshold])
    DD = np.zeros(len(threshold))
    win = 5
    
    for j,t in enumerate(threshold):
        temp = (pPart>=t)

        for w in range(win):
            temp2 = temp[1:]
            temp = temp[0:-1]*temp2
        if temp.max() == 0:
            DD[j] = len(temp)
        else:
            DD[j] = np.argmax(temp==1)
    
    
    return fpr,tpr,DD,threshold

def calCriteria(result,label):
    labelName = np.unique(label)
    nPart = result[label == labelName[0]]
    pPart = result[label == labelName[1]]
    fpr = nPart.mean() 
    tpr = pPart.mean()
    win = 5

    temp = pPart

    for w in range(win):
        temp2 = temp[1:]
        temp = temp[0:-1]*temp2
    if temp.max() == 0:
        DD = np.nan
    else:
        DD = np.argmax(temp==1)
    return fpr,tpr,DD
    
def threeResultAnd(result1,result2,result3,label,stepNum = 10):
    labelName = np.unique(label)
    
    threshold1 = genThreshold(result1,num=stepNum)
    threshold2 = genThreshold(result2,num=stepNum)
    threshold3 = genThreshold(result3,num=stepNum)


    fprVec = []
    tprVec = []
    DDVec = []

    for t1 in threshold1:
        result = result1>=t1
        if result.sum() == len(result1):
            fpr,tpr,DD = calCriteria(result,label)
            fprVec += [fpr]
            tprVec += [tpr]
            DDVec += [DD]
            continue
        for t2 in threshold2:
            result += (result2>=t2) 
            if result.sum() == len(result2):
                fpr,tpr,DD = calCriteria(result,label)
                fprVec += [fpr]
                tprVec += [tpr]
                DDVec += [DD]
                continue
            for t3 in threshold3:
                result += (result3>=t3) 
                fpr,tpr,DD = calCriteria(result,label)
                fprVec += [fpr]
                tprVec += [tpr]
                DDVec += [DD]

    return np.array(fprVec), np.array(tprVec),np.array(DDVec) 
                
def DetectPerfCVD(result,ft,step=0.01,stepNum=10):
    if len(result.shape) == 2:
        result = result.reshape(-1,3,1)
    label = np.zeros(len(result))
    label[ft:] = 1
    performanceMat = np.zeros((int(1/step),result.shape[-1],3))
    for i in range(result.shape[-1]):
        fpr,tpr,DD = threeResultAnd(result[:,0,i],result[:,1,i],\
                                    result[:,2,i],label,stepNum = stepNum)
            
        _,tpr = resample(fpr,tpr,step = step)
        fpr,DD = resample(fpr,DD,step = step)

        performanceMat[:,i,0] = fpr
        performanceMat[:,i,1] = tpr
        performanceMat[:,i,2] = DD/(len(result)-ft)
    return performanceMat

def twoResultAnd(result1,result2,label,stepNum = 10):
    labelName = np.unique(label)
    
    threshold1 = genThreshold(result1,num=stepNum)
    threshold2 = genThreshold(result2,num=stepNum)


    fprVec = []
    tprVec = []
    DDVec = []

    for t1 in threshold1:
        result = result1>=t1
        if result.sum() == len(result1):
            fpr,tpr,DD = calCriteria(result,label)
            fprVec += [fpr]
            tprVec += [tpr]
            DDVec += [DD]
            continue
        for t2 in threshold2:
            result += (result2>=t2) 
            fpr,tpr,DD = calCriteria(result,label)
            fprVec += [fpr]
            tprVec += [tpr]
            DDVec += [DD]
        
    return np.array(fprVec), np.array(tprVec),np.array(DDVec) 
                
def DetectPerfT2SPE(result,ft,step=0.01,stepNum=10):
    if len(result.shape) == 2:
        result = result.reshape(-1,2,1)
    label = np.zeros(len(result))
    label[ft:] = 1
    performanceMat = np.zeros((int(1/step),result.shape[-1],3))
    for i in range(result.shape[-1]):
        fpr,tpr,DD = twoResultAnd(result[:,0,i],result[:,1,i],\
                                    label,stepNum = stepNum)
            
        _,tpr = resample(fpr,tpr,step = step)
        fpr,DD = resample(fpr,DD,step = step)

        performanceMat[:,i,0] = fpr
        performanceMat[:,i,1] = tpr
        performanceMat[:,i,2] = DD/(len(result)-ft)
    return performanceMat    

def DetectPerf(result,ft,step=0.01,stepNum =10):
    if len(result.shape) == 1:
        result = result.reshape(-1,1)
    label = np.zeros(result.shape)
    label[ft:,:] = 1
    performanceMat = np.zeros((int(1/step)+1,result.shape[1],3))
    for i in range(result.shape[1]):
        fpr,tpr,DD,threshold = roc_curve2(label[:,i],result[:,i],stepNum=stepNum)
        #print(DD[0:3])
        
        _,tpr = resample(fpr,tpr,step = step)
        fpr,DD = resample(fpr,DD,step = step)

        performanceMat[:,i,0] = fpr
        performanceMat[:,i,1] = tpr
        performanceMat[:,i,2] = DD
    return performanceMat

def threeCriterio(result,th,ft,win=2,healthy=False):
    if len(result.shape) == 1:
        result = result.reshape(len(result),1)
        
    if healthy:
        FAR = np.mean(np.any(result[:ft,:] >=th,axis=1))
        return FAR
    else:
        i = 0
        DD = np.nan
        while i <= len(result)-win:
            temp = result[i:i+win,:] >= th
            if np.any(temp.sum(axis=0) == win) and i >= ft:
                DD = i-ft
                break
            i += 1
        FAR = np.mean(np.any(result[:ft,:] >=th,axis=1))
        MDR = np.mean(np.any(result[ft:,:] <th,axis=1))
        return FAR,MDR,DD
    
def  ConfusionMatrix(result,GT,m=None,CM=None):
    if CM is None:
        CM = np.zeros((m,m))
    if type(result) == int or type(result) == float:
        CM[int(GT),int(result)] +=1
    else:
        for i in range(len(result)):
            CM[int(GT),int(result[i])] +=1
    return CM

def sensitivity(label,value):
    try:
        r,c = label.shape
    except:
        c = len(label)
        r = 1
        label = label.reshape(1,c)
        value = value.reshape(1,c)
    
    labelValue = list(set(label[0]))
    CrBefore = value[label==labelValue[0]].reshape(r,-1)
    CrAfter = value[label==labelValue[1]].reshape(r,-1)
    
    CrBeforeMean = CrBefore.mean(axis=1)
    CrBeforeMax = CrBefore.max(axis=1)
    CrAfterMean = CrAfter.mean(axis=1)
    
    
    senValue = (CrAfterMean - CrBeforeMean ) /  (CrBeforeMax - CrBeforeMean )
        
        
    return senValue
    
    

