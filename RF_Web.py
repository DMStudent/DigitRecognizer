# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 22:56:58 2016

@author: DELL-PC
"""

# -*- coding: utf-8 -*- 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier 
#from sklearn import datasets    
#机器学习是对不同特征的学习，如果特征间相互独立，那么知道和不知道这些特征叫什么根本没有什么区别，
#都是某个维度的数字罢了， 但是很多时候关键就在于不同特征之间的关系，隐含特征的挖掘，也即Feature Engineering  
#数据路径 
trainPath='train.csv' 
testPath='test.csv'  #准备训练数据 
rawData=pd.read_csv(trainPath).values 
trainData=rawData[:,1:] 
trainLabel=rawData[:,0] 
testData= pd.read_csv(testPath).values 
X=trainData 
Y=trainLabel  #随机森林训练和测试 
clf=RandomForestClassifier(n_estimators=400, n_jobs=-1) 
clf=clf.fit(X,Y) 
testLabel=clf.predict(testData)  #训练结果保存 
df=pd.DataFrame({'ImageId':list(range(1,28001)), 'Label':testLabel}) 
df.to_csv('Result_RF.csv',header=True,index=False) 