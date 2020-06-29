#https://www.kaggle.com/mirichoi0218/insurance
import numpy as np
import pandas as pd

dataset = pd.read_csv("F:\MLudemy\MLAZ\\100DaysMLCode\Regression\insurance\insurance.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
# y = y.reshape(len(y),1)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 4, 5])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)

# y_pred = regressor.predict(X_test)
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# poly_reg = PolynomialFeatures(degree = 4)
# X_poly = poly_reg.fit_transform(X_train)
# regressor = LinearRegression()
# regressor.fit(X_poly, y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(poly_reg.transform(X_test))
# np.set_printoptions(precision=2)
# # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# y_train = sc_y.fit_transform(y_train)

# # Training the SVR model on the Training set
# from sklearn.svm import SVR
# regressor = SVR(kernel = 'rbf')
# regressor.fit(X_train, y_train.ravel())

# # Predicting the Test set results
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(X_test)))
# np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Training the Random Forest Regression model on the whole dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

def checkPerformance(y_test, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_test, y_pred)

score = checkPerformance(y_test, y_pred)
print(score)