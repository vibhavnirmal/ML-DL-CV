import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

training_data = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Kaggle\\titanic\\train.csv")
testing_data = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Kaggle\\titanic\\test.csv")

# null_columns=training_data.columns[training_data.isnull().any()]
# print(training_data[null_columns].isnull().sum())

features = ["Pclass","Sex","SibSp","Parch"]

X_train = pd.get_dummies(training_data[features])
y_train = training_data.iloc[:, [1]].values

X_test = pd.get_dummies(testing_data[features])

def logisReg():
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def knn():
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def nbcla():
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def dtcla():
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def rfcla():
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def svm():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def ksvm():
    from sklearn.svm import SVC
    classifier = SVC(kernel="", random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    return results(classifier)

def xgboost():
    from xgboost import XGBClassifier
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return results(classifier)

def catboost():
    from catboost import CatBoostClassifier
    classifier = CatBoostClassifier()
    classifier.fit(X_train, y_train)
    return results(classifier)
    
def results(classifier):
    predictions = classifier.predict(X_test)
    return predictions
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # y_pred = classifier.predict(X_test)
    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)
    # accuracy = accuracy_score(y_test, y_pred)
    # return accuracy

def gridSearch():
    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf', random_state = 0)
    classifier.fit(X_train, y_train.ravel())
    from sklearn.model_selection import GridSearchCV
    parameters = [{'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'kernel': ['linear']},
                {'C': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
    grid_search = GridSearchCV(estimator = classifier,
                            param_grid = parameters,
                            scoring = 'accuracy',
                            cv = 10,
                            n_jobs = -1)
    grid_search.fit(X_train, y_train.ravel())
    return results(grid_search)

#removed model for ann1.csv as accuracy was lower
#These model is of ann2.csv
def ann():
    import tensorflow as tf
    ann = tf.keras.models.Sequential()

    # Adding the input layer and the first hidden layer
    ann.add(tf.keras.layers.Dense(units=8, input_shape=(5,), activation='relu'))
    # Adding the second hidden layer
    ann.add(tf.keras.layers.Dense(units=15, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=8, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
    # Adding the output layer
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    # Part 3 - Training the ANN
    # Compiling the ANN
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Training the ANN on the Training set
    ann.fit(X_train, y_train, batch_size = 32, epochs = 1000)
    
    result = []
    Y_pred = ann.predict(np.asarray(X_test).astype(np.float32))
    Y_pred = (Y_pred>0.51)

    for i in range(len(Y_pred)):
        if Y_pred[i][0] == True :
            result.append(1)
        else :
            result.append(0)
    
    # y_pred = np.empty_like(predx)
    # predictions = classifier.predict(X_test)
    # for i in predictions:
        # if i[0] >= 0.5:
            # np.append(y_pred, [1])
        # elif i[0] < 0.5:
            # np.append(y_pred, [0])

    return result

pred = ann()
print(pred)

PassengerId = testing_data["PassengerId"] # PassengerId,Survived
SurvivedResult = pd.DataFrame({'Survived': pred})
results = pd.concat([PassengerId,SurvivedResult],axis=1)
results.to_csv("F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_ann.csv",sep = ',',index=False)

# print("Logistic Regression Score = ", logisReg())
# print("KNN Score = ", knn())
# print("Naive Bayes Score = ", nbcla())
# print("Desicion Tree Classification Score = ", dtcla())
# print("Random Forest Classification Score = ", rfcla())
# print("Support Vector Machine Score = ", svm())
# print("Kernel Support Vector Machine Score = ", ksvm())
# print("Catboost Score = ", catboost())

# output1 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': logisReg()})
# output1.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_logisticReg.csv', index=False)

# output2 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': knn()})
# output2.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_knn.csv', index=False)

# output3 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': nbcla()})
# output3.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\\_nbcla.csv', index=False)

# output4 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': dtcla()})
# output4.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_dtcla.csv', index=False)

# output5 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': rfcla()})
# output5.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\\_rfcla.csv', index=False)

# output5 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': gridSearch()})
# output5.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\\_gridSearch.csv', index=False)

# output6 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': svm()})
# output6.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_svm.csv', index=False)

# output7 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': ksvm()})
# output7.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_ksvm.csv', index=False)

# output8 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': xgboost()})
# output8.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_xgboost.csv', index=False)

# output9 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': catboost()})
# output9.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_catboost.csv', index=False)

# output10 = pd.DataFrame({'PassengerId': testing_data.PassengerId, 'Survived': pred})
# output10.to_csv('F:\MLudemy\MLAZ\\100DaysMLCode\Kaggle\\titanic\_ann.csv', index=False)