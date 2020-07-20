import pandas as pd
import numpy as np

dataset = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Classification\iris\\bezdekIris.csv")

X = dataset.iloc[:,0:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 32)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(max_iter=3000)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(y_pred, y_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)*100)

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy:{:.2f} %".format(accuracy.mean()*100))
print("Standard Deviation:{:.2f} %".format(accuracy.std()*100))