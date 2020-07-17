import numpy as np
import pandas as pd
import collections
from matplotlib import pyplot as plt

training_set = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\train.txt")
tesda = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\test.txt")
tesda_train = tesda.iloc[:, 1:-1].values
tesda_test = tesda.iloc[:, -1].values

X = training_set.iloc[:, 1:-1].values
y = training_set.iloc[:, -1].values

def mode(num_list):
    data = collections.Counter(num_list)
    data_list = dict(data)

    max_value = max(list(data.values()))
    mode_val = [num for num, freq in data_list.items() if freq == max_value]
    if len(mode_val) == len(num_list):
        return None
    else:
        return ', '.join(map(str, mode_val))

# Training Set plot
# from matplotlib.colors import ListedColormap
# for i, j in enumerate(np.unique(y)):
#     plt.scatter(X[y==j,0], X[y==j,1], c = ListedColormap(['coral', '#7CAE00', '#C77CFF', '#00BFC4'])(i), label = j)
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()

# # Testing Set plot
# plt.scatter(tesda_train[:,0], tesda_train[:,1])
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# ______GNB__________________________________________________________
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X ,y)
gnbPred = gnb.predict(tesda_train)
# ______NORMAL KNN__________________________________________________________
from sklearn.neighbors import KNeighborsClassifier
knnc = KNeighborsClassifier(n_neighbors=1, algorithm='auto', metric='minkowski', n_jobs=-1)
knnc.fit(X, y)
knnPred = knnc.predict(tesda_train)
# ______NORMAL SVC_________________________________________________________
from sklearn.svm import SVC
svcl = SVC(kernel='rbf')
svcl.fit(X, y)
svcPred = svcl.predict(tesda_train)
# ______NORMAL SVC_________________________________________________________
from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators=100, criterion='entropy')
rfcl.fit(X, y)
rfPred = rfcl.predict(tesda_train)
# _____LogisticReg_________________________________________________________
from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(solver='newton-cg')
logr.fit(X, y)
logregPred = logr.predict(tesda_train)
# _____XGBoost_____________________________________________________________
from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(X, y)
xgBoostPred = xgbc.predict(tesda_train)
# _____catBoost____________________________________________________________
from catboost import CatBoostClassifier
cbc = CatBoostClassifier(iterations=100, learning_rate=0.3)
cbc.fit(X, y)
catBoostPred = cbc.predict(tesda_train)
# _________________________________________________________________________

# print(mode([knnPred[5],catBoostPred[5][0],xgBoostPred[5],svcPred[5],logregPred[5]]))

# __________Manual method for max voting______________
final_pred0 = np.array([])
final_pred1 = np.array([])
final_pred2 = np.array([])
final_pred3 = np.array([])
for i, j in enumerate(tesda_train):
    final_pred0 = np.append(final_pred0, mode([knnPred[i],catBoostPred[i][0],xgBoostPred[i],svcPred[i],logregPred[i]]))
    final_pred1 = np.append(final_pred1, mode([knnPred[i],gnbPred[i],catBoostPred[i][0],xgBoostPred[i],svcPred[i],logregPred[i]]))
    final_pred2 = np.append(final_pred2, mode([knnPred[i],gnbPred[i],xgBoostPred[i],svcPred[i],logregPred[i]]))
    final_pred3 = np.append(final_pred3, mode([knnPred[i],rfPred[i],gnbPred[i],xgBoostPred[i],svcPred[i],logregPred[i]]))

from sklearn.metrics import accuracy_score
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred0)*100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred1)*100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred2)*100))
print("accuracy_score : {:.2f} %\n".format(accuracy_score(tesda_test, final_pred3)*100))
# 81.08 % without NaiveBayes
# 72.97 % with catboost + NaiveBayes
# 67.57 % without catboost
# 70.27 % with randomForest
# _________________________________________________________________________


# __________Using Voting Classifier______________
v_knnc = KNeighborsClassifier(n_neighbors=1, algorithm='auto', metric='minkowski')
v_svcl = SVC(kernel='rbf')
v_logr = LogisticRegression(solver='newton-cg')
v_xgbc = XGBClassifier()
v_gnb = GaussianNB()
v_rf = RandomForestClassifier(n_estimators=100, criterion='entropy')

from sklearn.ensemble import VotingClassifier

model1 = VotingClassifier(estimators=[('knn',v_knnc),('svc',v_svcl),('lr',v_logr),('xgb',v_xgbc)], voting='hard')
model1.fit(X, y)
v_pred1 = model1.predict(tesda_train)

model2 = VotingClassifier(estimators=[('knn',v_knnc),('svc',v_svcl),('gnb',v_gnb),('lr',v_logr),('xgb',v_xgbc)], voting='hard')
model2.fit(X, y)
v_pred2 = model2.predict(tesda_train)

model3 = VotingClassifier(estimators=[('knn',v_knnc),('svc',v_svcl),('gnb',v_rf),('lr',v_logr),('xgb',v_xgbc)], voting='hard')
model3.fit(X, y)
v_pred3 = model3.predict(tesda_train)

print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, v_pred1)*100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, v_pred2)*100))
print("accuracy_score : {:.2f} %\n".format(accuracy_score(tesda_test, v_pred3)*100))
# not working with catboost
# 78.38 % without gb
# 70.27 % with gb
# 81.08 % with random forest


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

# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import VotingClassifier, BaggingClassifier
# from sklearn.model_selection import KFold, cross_val_score

# kfold = KFold(n_splits=12)
# # create the sub models
# estimators = []
# model1 = LogisticRegression()
# estimators.append(('logistic', model1))
# model2 = DecisionTreeClassifier()
# estimators.append(('cart', model2))
# model3 = SVC()
# estimators.append(('svm', model3))
# # create the ensemble model
# ensemble = VotingClassifier(estimators, voting='hard')
# # es = BaggingClassifier(model3, max_samples=1.0, max_features=1.0)
# results = cross_val_score(ensemble, X, y, cv=kfold)
# # ress = cross_val_score(es, X, y, cv=kfold)
# print(results.mean())
# print(ress.mean())