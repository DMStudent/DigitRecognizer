# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:46:06 2016

@author: DELL-PC
"""

import numpy as np
import pandas as pd
import keras.utils.np_utils as kutils

img_rows, img_cols = 28, 28
  
if __name__ == '__main__':
    print ('read data!')
    dataset = pd.read_csv("../../Data/train.csv") # 注意自己数据路径
    trainData = dataset.iloc[:,1:].values
    labelsData = dataset.iloc[:,:1].values  
    tests = pd.read_csv("../../Data/test.csv") # 注意自己数据路径
    testData = tests.iloc[:,:].values
    
    test = testData.reshape(testData.shape[0], 1, img_rows, img_cols)
    test = test.astype(float)
    test /= 255.0
    
    train = trainData.reshape(trainData.shape[0], 1, img_rows, img_cols)
    train = train.astype(float)
    train /= 255.0
    labels = kutils.to_categorical(labelsData)
                
    np.save('train.npy', train)
    np.save('labels.npy', labels)
    np.save('test.npy', test)
#    train = np.load('train01.npy')
#    labels = np.load('labels.npy')
#    test = np.load('test01.npy')
    
    
    print ('done reading test!')  
