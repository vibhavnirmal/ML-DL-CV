# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 23:59:24 2018

@author: Vibhav K Nirmal
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("trafficstop.csv")

dataset = dataset.replace(np.nan,4.0)

X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,10].values

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = "NaN",strategy=)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,
                                                    random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_train)

import statsmodels.formula.api as sm

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:,[0,1,2,3,4,5,6,7,9]]
regressor_ols = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_ols.summary()

# Splitting the dataset into the Training set and Test set
X_opt_train, X_opt_test, Y_opt_train, Y_opt_test = train_test_split(X_opt, Y,
                                                                    test_size = 1/3,
                                                                    random_state = 0)
regressor_opt = LinearRegression()
regressor_opt.fit(X_opt_train, Y_opt_train)
 
y_opt_pred = regressor_opt.predict(X_opt_test)

plt.scatter(X_opt_train, Y_opt_train)