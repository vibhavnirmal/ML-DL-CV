import collections

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

training_set = pd.read_csv(
    "F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\train.txt"
)
tesda = pd.read_csv(
    "F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\dart\\test.txt"
)
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
    return ", ".join(map(str, mode_val))


# Training Set plot
from matplotlib.colors import ListedColormap
for i, j in enumerate(np.unique(y)):
    plt.scatter(X[y==j,0], X[y==j,1], c = ListedColormap(['coral', '#7CAE00', '#C77CFF', '#00BFC4'])(i), label = j)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Testing Set plot
plt.scatter(tesda_train[:,0], tesda_train[:,1])
plt.xlabel("x")
plt.ylabel("y")
plt.show()

gnb = GaussianNB()
gnb.fit(X, y)
gnbPred = gnb.predict(tesda_train)

knnc = KNeighborsClassifier(n_neighbors=1, algorithm="auto", metric="minkowski", n_jobs=-1)
knnc.fit(X, y)
knnPred = knnc.predict(tesda_train)

svcl = SVC(kernel="rbf")
svcl.fit(X, y)
svcPred = svcl.predict(tesda_train)

rfcl = RandomForestClassifier(n_estimators=100, criterion="entropy")
rfcl.fit(X, y)
rfPred = rfcl.predict(tesda_train)

logr = LogisticRegression(solver="newton-cg")
logr.fit(X, y)
logregPred = logr.predict(tesda_train)

xgbc = XGBClassifier()
xgbc.fit(X, y)
xgBoostPred = xgbc.predict(tesda_train)

cbc = CatBoostClassifier(iterations=100, learning_rate=0.3)
cbc.fit(X, y)
catBoostPred = cbc.predict(tesda_train)

# __________Manual method for max voting______________
final_pred0 = np.array([])
final_pred1 = np.array([])
final_pred2 = np.array([])
final_pred3 = np.array([])
for i, j in enumerate(tesda_train):
    final_pred0 = np.append(final_pred0,
                    mode(
                        [knnPred[i], catBoostPred[i][0], xgBoostPred[i], svcPred[i], logregPred[i]]
                        ))
    final_pred1 = np.append(final_pred1,
                    mode(
                        [knnPred[i], gnbPred[i], catBoostPred[i][0], xgBoostPred[i], svcPred[i], logregPred[i]]
                        ))
    final_pred2 = np.append(final_pred2,
                    mode(
                        [knnPred[i], gnbPred[i], xgBoostPred[i], svcPred[i], logregPred[i]]
                        ))
    final_pred3 = np.append(final_pred3,
                    mode(
                        [knnPred[i], rfPred[i], gnbPred[i], xgBoostPred[i], svcPred[i], logregPred[i]]
                        ))


print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred0) * 100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred1) * 100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred2) * 100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, final_pred3) * 100))
print("\n")
"""
81.08 % without NaiveBayes
72.97 % with catboost + NaiveBayes
67.57 % without catboost
70.27 % with randomForest
"""


v_knnc = KNeighborsClassifier(n_neighbors=1, algorithm="auto", metric="minkowski")
v_svcl = SVC(kernel="rbf")
v_logr = LogisticRegression(solver="newton-cg")
v_xgbc = XGBClassifier()
v_gnb = GaussianNB()
v_rf = RandomForestClassifier(n_estimators=100, criterion="entropy")


model1 = VotingClassifier(
    estimators=[("knn", v_knnc), ("svc", v_svcl), ("lr", v_logr), ("xgb", v_xgbc)],
    voting="hard",
)
model1.fit(X, y)
v_pred1 = model1.predict(tesda_train)

model2 = VotingClassifier(
    estimators=[
        ("knn", v_knnc),
        ("svc", v_svcl),
        ("gnb", v_gnb),
        ("lr", v_logr),
        ("xgb", v_xgbc),
    ],
    voting="hard",
)
model2.fit(X, y)
v_pred2 = model2.predict(tesda_train)

model3 = VotingClassifier(
    estimators=[
        ("knn", v_knnc),
        ("svc", v_svcl),
        ("gnb", v_rf),
        ("lr", v_logr),
        ("xgb", v_xgbc),
    ],
    voting="hard",
)
model3.fit(X, y)
v_pred3 = model3.predict(tesda_train)

print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, v_pred1) * 100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, v_pred2) * 100))
print("accuracy_score : {:.2f} %".format(accuracy_score(tesda_test, v_pred3) * 100))
print("\n")
"""
not working with catboost
78.38 % without gb
70.27 % with gb
81.08 % with random forest
"""
