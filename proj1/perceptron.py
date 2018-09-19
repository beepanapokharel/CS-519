import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Perceptron ( object ) :
    def __init__(self , eta=0.1, n_iter=10, random_state=1):
        self.eta = eta
        self . n_iter = n_iter
        self.random_state = random_state

        
    def predict ( self , X) :
        z = np.dot(X, self.w_[1:]) + self.w_[0] 
        return np.where(z >= 0.0, 1, -1)
    
    def learn(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors=[] #wrong classifications
        
        for i in range( self.n_iter ):
            errors = 0
            for xi, target in zip(X, y):
                yii=self.predict(xi)
                update = self.eta * (target - yii)
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors+=np.where(target==yii,0,1)
                if(errors<0):
                    return self
            self.errors.append(errors)
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
        self.accuracy=(c/(c+mc))*100
        return self
        
        
