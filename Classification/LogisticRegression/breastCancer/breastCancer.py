import pandas as pd
import numpy as np

missing_values = ["?"]
# dataset = pd.read_csv('breast_cancer.csv')
dataset = pd.read_csv('breast-cancer-wisconsin.csv', na_values = missing_values)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# dropping rows with missing values
dataset.dropna(inplace=True)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imputer.fit(X[:, 1:-1])
X[:, 1:-1] = imputer.transform(X[:, 1:-1])
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred)*100)

from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy:{:.2f} %".format(accuracy.mean()*100))
print("Standard Deviation:{:.2f} %".format(accuracy.std()*100))