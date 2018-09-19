import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class SGD(object):
    def __init__(self , eta=0.1, n_iter=10, random_state=1):
        self.eta = eta
        self . n_iter = n_iter
        self.random_state = random_state
       
    def predict ( self , X) :
        z = np.dot(X, self.w_[1:]) + self.w_[0] 
        return np.where(z >= 0.0, 1, -1)
    
    def shuffle(self , X, y):
        r = np.random.permutation(y)
        return X[r], y[r]

    def learn(self,X,y):
        self.w_ = np.random.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost=[]
        
        for i in range(self.n_iter):
            X,y = self.shuffle(X,y)
            cost=[]
            for xi,target in zip(X,y):
                cost.append(self.update_weights(xi , target))
            avg_cost = sum(cost) / len(y)
            self .cost.append(avg_cost)
        return self
        
        
    def update_weights(self,X,y):
        net_input=np.dot(X,self.w_[1:]+self.w_[0])
        error=y-net_input
        self.w_[0]+=self.eta*error
        self.w_[1:]+=self.eta*X.dot(error)
        cost=1./2*(error**2)
        return cost
    
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
        self.accuracy=(c/(c+mc))*100
        return self




