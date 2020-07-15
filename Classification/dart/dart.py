import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

trada = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\train.txt")
tesda = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\test.txt")
tesda_train = tesda.iloc[:, 1:-1].values
tesda_test = tesda.iloc[:, -1].values

X = trada.iloc[:, 1:-1].values
y = trada.iloc[:, -1].values

# from matplotlib.colors import ListedColormap
# for i, j in enumerate(np.unique(y)):
#     plt.scatter(X[y==j,0], X[y==j,1], c = ListedColormap(['coral', '#7CAE00', '#C77CFF', '#00BFC4'])(i), label = j)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# plt.scatter(tesda_train[:,0], tesda_train[:,1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# ______NORMAL KNN__________________________________________________________
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier(n_neighbors=1, algorithm='auto', metric='minkowski', n_jobs=-1)
# classifier.fit(X, y)

# tesda_pred = classifier.predict(tesda_train)
# print(np.concatenate((tesda_pred.reshape(len(tesda_pred),1), tesda_test.reshape(len(tesda_test),1)),1))

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(tesda_test, tesda_pred)
# print(cm)
# print("{:.2f} %".format(accuracy_score(tesda_test, tesda_pred)*100))
# _________________________________________________________________________


# ____CROSS VALIDATION_____________________________________________________
# from sklearn.neighbors import KNeighborsClassifier
# knnc = KNeighborsClassifier(n_neighbors=1)
# from sklearn.model_selection import cross_val_score
# cvScore = cross_val_score(knnc, X, y, cv=12)
# # print(cvScore)
# print('cvScore mean:{}'.format(np.mean(cvScore)))
# _________________________________________________________________________


from sklearn.metrics import accuracy_score
# _________________________________________________________________________
# print("\n______KNN______")
# """ K Nearest Neighbors """
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# knn = KNeighborsClassifier()
# param_grid = {'n_neighbors': np.arange(1, 25)}
# knn_gscv = GridSearchCV(knn, param_grid, cv=10)
# knn_gscv.fit(X, y)
# print(knn_gscv.best_params_) # to check top performing n_neighbors value
# print("best_score(on training): "+str(knn_gscv.best_score_*100)) # to check mean score for the top performing value of n_neighbors
# # 70%
# tesda_predk = knn_gscv.predict(tesda_train)
# print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, tesda_predk)*100))
# _________________________________________________________________________


# _________________________________________________________________________
# print("\n______SVC______")
# """ Support Vector Machine """
# from sklearn.svm import SVC
# from sklearn.model_selection import GridSearchCV
# svcl = SVC()
# param_grid = {'C': np.arange(1, 25)}
# svc_gscv = GridSearchCV(svcl, param_grid)
# svc_gscv.fit(X, y)
# print(svc_gscv.best_params_) # to check top performing n_neighbors value
# print("best_score(on training): "+str(svc_gscv.best_score_*100)) # to check mean score for the top performing value of n_neighbors
# # 74%
# tesda_preds = svc_gscv.predict(tesda_train)
# print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, tesda_preds)*100))
# print("____________")
# _________________________________________________________________________


# print("Stacking (Meta Ensembling)")

# from sklearn.ensemble import BaggingClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# bagging = BaggingClassifier()
# param_grid = {'base_estimator':[KNeighborsClassifier()],'max_samples': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 'max_features': [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
# bag_gscv = GridSearchCV(bagging, param_grid, cv=10)
# bag_gscv.fit(X, y)
# tesda_predb = bag_gscv.predict(tesda_train)
# print(bag_gscv.best_params_)
# print("best_score(on training): "+str(bag_gscv.best_score_*100))
# print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, tesda_predb)*100))
# print("____________")


# Voting Ensemble for Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, BaggingClassifier
from sklearn.model_selection import KFold, cross_val_score

kfold = KFold(n_splits=12)
# create the sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))
# create the ensemble model
ensemble = VotingClassifier(estimators, voting='hard')
# es = BaggingClassifier(model3, max_samples=1.0, max_features=1.0)
results = cross_val_score(ensemble, X, y, cv=kfold)
# ress = cross_val_score(es, X, y, cv=kfold)
print(results.mean())
# print(ress.mean())


# from xgboost import XGBClassifier
# xgbc = XGBClassifier()
# xgbc.fit(X, y)

# tesda_predx = xgbc.predict(tesda_train)
# print("{:.2f} %".format(accuracy_score(tesda_test, tesda_predx)*100))

# from catboost import CatBoostClassifier
# cbc = CatBoostClassifier(iterations=100, learning_rate=0.3)
# cbc.fit(X, y)
# tesda_predc = cbc.predict(tesda_train)
# print("{:.2f} %".format(accuracy_score(tesda_test, tesda_predc)*100))