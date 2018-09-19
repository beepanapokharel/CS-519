import numpy as np
import pandas as pd
import sys
import time

from matplotlib import pyplot as plt
start_time = time.time()
def main():
    classifier_name=sys.argv[1]
    datapath=sys.argv[2]
    df=pd.read_csv(datapath)
    X=df.iloc[0:100,[0,2]].values
    y=df.iloc[0:100,4].values
    y=np.where(y=='Iris-setosa',1,-1)
    
    if(classifier_name=='Perceptron'):
        from perceptron import Perceptron
        model=Perceptron(eta=0.01,n_iter=10) 
        model.learn(X,y)
        print('errors for this classification are:\n',model.errors)
        plt.plot(range(1,len(model.errors)+1), model.errors,marker='o')
        model.testdatairis('test.csv')
        print("Accuracy of Perceptron is:",model.accuracy)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    elif(classifier_name=='Adaline'):
        from adaline import Adaline
        model=Adaline(eta=0.01,n_iter=20)
        X_std = np.copy(X)
        X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
        X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
        model.learn(X_std,y)
        print('sum of errors in each iteration for this classification are:\n',model.cost)
        plt.plot(range(1,len(model.cost)+1), model.cost,marker='o')
        model.testdatairis('test.csv')
        print("Accuracy of Adaline is:",model.accuracy)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    elif(classifier_name=='SGD'):
        from sgd import SGD
        model=SGD(eta=0.01,n_iter=15)
        model.learn(X,y)
        print('sum of errors in each interation for this classification are:\n' ,model.cost)
        plt.plot(range(1,len(model.cost)+1), model.cost,marker='o')
        model.testdatairis('test.csv')
        print("Accuracy of SGD is:",model.accuracy)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    elif(classifier_name=='Onevsrest'):
        from onevsrest import Onevsrest
        model=Onevsrest(eta=0.01,n_iter=15)
        model.learn(X,y)
        print('sum of errors in each interation for this classification are:\n' ,model.cost)
        plt.plot(range(1,len(model.cost)+1), model.cost,marker='o')
        model.testdatairis('test1.csv')
        print("--- %s seconds ---" % (time.time() - start_time))
        
        
    else:
        print("invalid classifier")
        return

    plt.title(classifier_name)
    plt.xlabel('iteration')
    plt.ylabel('errors')
    plt.show()
    return(model)



if __name__ == "__main__":
    
    main()
    
