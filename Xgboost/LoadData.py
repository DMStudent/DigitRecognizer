# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 21:46:06 2016

@author: DELL-PC
"""

import numpy as np
import csv
import pandas as pd
    
if __name__ == '__main__':
    print ('read data!')
    dataset = pd.read_csv("../../Data/train.csv") # 注意自己数据路径
    train = dataset.iloc[:,1:].values
    labels = dataset.iloc[:,:1].values  
    tests = pd.read_csv("../../Data/test.csv") # 注意自己数据路径
    test = tests.iloc[:,:].values
    
    [row, col] = np.shape(train);
    for i in range(row):
        for j in range(col):
            if train[i,j] > 0:
                train[i,j] = 1
    
    
    [row, col] = np.shape(test);
    for i in range(row):
        for j in range(col):
            if test[i,j] > 0:
                test[i,j] = 1
                
    np.save('train01.npy', train)
    np.save('labels.npy', labels)
    np.save('test01.npy', test)
    train = np.load('train01.npy')
    labels = np.load('labels.npy')
    test = np.load('test01.npy')
    
    
    print ('done reading test!')  
