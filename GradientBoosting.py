import time
import pandas as pd 
from sklearn.ensemble import GradientBoostingClassifier   
#机器学习是对不同特征的学习，如果特征间相互独立，那么知道和不知道这些特征叫什么根本没有什么区别，
#都是某个维度的数字罢了， 但是很多时候关键就在于不同特征之间的关系，隐含特征的挖掘，也即Feature Engineering  
#数据路径 

if __name__ == "__main__":
    t1 = time.time()
    print ("loading data...")
    trainPath='train.csv' 
    testPath='test.csv'  #准备训练数据 
    rawData=pd.read_csv(trainPath).values 
    trainData=rawData[:,1:] 
    trainLabel=rawData[:,0] 
    testData= pd.read_csv(testPath).values 
    X=trainData 
    Y=trainLabel  #随机森林训练和测试 
    print ("start training GradientBoosting...")
    gbClf = GradientBoostingClassifier(learning_rate=1, n_estimators=200)       # params: by default
    gbClf.fit(X,Y)
    print ("Start classify test set...")
    testLabel = gbClf.predict(testData)
    
    df=pd.DataFrame({'ImageId':list(range(1,28001)), 'Label':testLabel}) 
    df.to_csv('Result_GBDT.csv',header=True,index=False) 
    t2 = time.time()
    print ("Done! It cost",t2-t1,"s")