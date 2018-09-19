import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Adaline(object):
    def __init__(self , eta=0.1, n_iter=10, random_state=1):
        self.eta = eta
        self . n_iter = n_iter
        self.random_state = random_state
        
    def predict ( self , X) :
        z = np.dot(X, self.w_[1:]) + self.w_[0] 
        return np.where(z >= 0.0, 1, -1)
    
    def learn(self,X,y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
       
        self.cost=[]
        self.errors=[]
        
        for i in range(self.n_iter):
            net_input=np.dot(X,self.w_[1:]+self.w_[0])
            errors=y-net_input
            self.w_[0]+=self.eta*errors.sum()
            self.w_[1:]+=self.eta*X.T.dot(errors)
            
            cost=1.0/2*sum(errors**2)
            self.cost.append(cost)
        
        return self
    
    def testdatairis(self,datapath):
        self.accuracy=0
        df=pd.read_csv(datapath)
        X1=df.iloc[0:50,[0,2]].values
        y1=df.iloc[0:50,4].values
        mc=0
        c=0
        for i, target in zip(X1, y1):
            if(self.predict(i)==target):
                c=c+1
            else:
                mc=mc+1
        self.accuracy=c/(c+mc)
        return self







