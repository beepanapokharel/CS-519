from sklearn . preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import time
import sys
from sklearn . linear_model import LinearRegression
from sklearn . linear_model import RANSACRegressor
from sklearn . preprocessing import PolynomialFeatures
from sklearn . linear_model import Ridge
from sklearn . linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

start_time=time.time()

def main():
    regression_name=sys.argv[1]
    datapath=sys.argv[2]

    if(datapath=='housing.data.txt'):
        df = pd.read_csv('housing.data.txt',
                 header=None,
                 sep='\s+')

        df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS',
              'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

        X=df.iloc[:,:-1]
        y=df['MEDV'].values

    else:
        df=pd.read_csv('all_breakdown.csv')
        df=df.fillna(0)
        X=df.iloc[:,1:-1]
        y=df['WIND TOTAL'].values

    y2d=y[ : , np . newaxis ] #change one dimensional array to two dimensions
    sc_x = StandardScaler ( )
    sc_y = StandardScaler ( )
    sc_x.fit (X)
    sc_y.fit (y)
    x_std = sc_x.transform(X)
    y_std = sc_y.transform(y2d).flatten()

    X_train, X_test, y_train, y_test = train_test_split(x_std, y_std, test_size=0.3, random_state=0)
    if (regression_name=="Linear"):
        model = LinearRegression ()
        model.fit (X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        print('Linear Regression')
        print ( 'Slope : %.3f ' %model.coef_[0])
        print ( 'Intercept : %.3f' %model.intercept_)
        print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)));

    elif (regression_name=="RANSAC"):
        ransac = RANSACRegressor(LinearRegression() , max_trials=100, min_samples=50, loss='absolute_loss' , residual_threshold=5.0, random_state=1)
        ransac. fit (X_train,y_train)
        print('RANSAC Regressor')
        print ( 'Slope : %.3f ' %ransac.estimator_.coef_[0])
        print ( 'Intercept : %.3f' %ransac.estimator_.intercept_)
#        print( 'Score of the prediction: %.3f' %ransac.score(X_test,y_test))
        y_train_pred = ransac.predict(X_train)
        y_test_pred = ransac.predict(X_test)
        print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)));

    elif (regression_name=="Ridge"):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        y_train_pred = ridge.predict(X_train)
        y_test_pred = ridge.predict(X_test)
        print('Ridge Regularization')
        print ('Slope : %.3f'%ridge.coef_[0])
        print ('Intercept : %.3f' %ridge.intercept_)
        print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)));

    elif (regression_name=="Lasso"):
        lasso = Lasso(alpha=1.0)
        lasso.fit(X_train, y_train)
        y_train_pred = lasso.predict(X_train)
        y_test_pred = lasso.predict(X_test)
        print('Lasso Regularization')
        print ( 'Slope : %.3f ' %lasso.coef_[0])
        print ( 'Intercept : %.3f' %lasso.intercept_ )

        print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)));

    elif (regression_name=="Nonlinear"):
        tree = DecisionTreeRegressor(max_depth=3) 
        tree. fit (X_train,y_train)
        y_test_pred = tree . predict (X_test)
        y_train_pred = tree.predict(X_train)
        print('Non linear Regression - Decision Tree Regressor')
        print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)));

    elif (regression_name=="Normal"):
        if(datapath=='housing.data.txt'):
            onevec = np.ones((X_train.shape[0]) ) #this generates a 1−dimensional array 
            onevec = onevec [ : , np . newaxis ] # changes the 1−dimensional array to 2−dimensional array 
            Xb = np.hstack((onevec,X_train)) # Xb is a 2−dimensional array 
            w = np.zeros(X_train.shape[1])
            z = np.linalg.inv(np.dot(Xb.T,Xb))
            w = np.dot(z, np.dot(Xb.T,y_train))
            print('Normal Equation Solution')
            print('Slope: %.3f' %w[1])
            print ( 'Intercept : %.3f' %w[0]);
            yhat = np.dot(Xb,w.T)
            print('MSE train: %.3f,' %mean_squared_error(y_train, yhat))
        else:
            print('Not Applicable');
    else:
        print ("No regression available with the given name");


    print("--- Time taken is %s seconds ---" % (time.time() - start_time))

if __name__=="__main__":
        main()
