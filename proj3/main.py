from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import time
import pandas as pd
import sys
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn . neighbors import KNeighborsClassifier
from sklearn . linear_model import LogisticRegression

start_time=time.time()
def main():
    classifier_name=sys.argv[1]
    datapath=sys.argv[2]
    if(datapath=='digits'):
        digits1 = datasets.load_digits()
        X=digits1.data[:,[2,3]]
        y=digits1.target
    else:
        df=pd.read_csv('subject1_self.log', sep='\t', lineterminator='\n')
        X=df.iloc[:, 2:3]
        Y=df.iloc[:,-1]
        y1=Y.values
        y=[]
        for i in range(len(df)):
            y.append(y1[i])

    sc = StandardScaler()
    sc.fit(X)
    X_std = sc.transform(X)
    X_tr,X_ts,y_tr,y_ts=train_test_split(X_std,y,test_size=0.3,random_state=1,stratify=y)
    if(classifier_name=="Perceptron"):
        cn=Perceptron(n_iter=40,eta0=0.1, random_state=1);

    elif(classifier_name=="LinearSVM"):
    	cn=SGDClassifier(loss='hinge');

    elif(classifier_name=="NonlinearSVM"):
        cn=SVC(kernel="rbf", random_state=1, gamma=0.2, C=1.0);

    elif(classifier_name=="DecisionTree"):
        cn=DecisionTreeClassifier ( criterion= "gini" ,max_depth=4,random_state=1);

    elif(classifier_name=="KNN"):
        cn=KNeighborsClassifier(n_neighbors=5, p=2, metric="minkowski");

    elif(classifier_name=="LogRegression"):
        cn=LogisticRegression(C=100.0, random_state=1)

    else:
    	cn=Perceptron(n_iter=40,eta0=0.1, random_state=1);

    cn.fit(X_tr, y_tr)
    y_pred=cn. predict (X_ts)
    error = (y_ts!=y_pred).sum()
    print( "Misclassified samples: %d" %error)
    print ( "Accuracy: %0.2f " %cn.score(X_ts,y_ts))
    print("--- Time taken is %s seconds ---" % (time.time() - start_time))


if __name__=="__main__":
	main()
