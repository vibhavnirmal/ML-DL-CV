import numpy as np
import pandas as pd

class Regression:
    def __init__(self, path, test_size=0.2, random_state=0):
        dataset = pd.read_csv(path)
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values

        self.splitDataset(X, y, dataset, test_size, random_state)

        # from sklearn.preprocessing import StandardScaler
        # sc = StandardScaler()
        # self.X_train[:,:] = sc.fit_transform(self.X_train[:,:])
        # self.X_test[:,:] = sc.transform(self.X_test[:,:])
        # print(self.X_test)

    def splitDataset(self, X, y, dataset, test_size, random_state):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Evaluating the Model Performance
    def checkPerformance(self, y_test, y_pred):
        from sklearn.metrics import r2_score
        return r2_score(y_test, y_pred)

    #Simple Linear Regression
    def simpleReg(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

        score = self.checkPerformance(self.y_test, y_pred)
        return score

    #Multiple Linear Regression
    def multiReg(self):
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)

        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

        score = self.checkPerformance(self.y_test, y_pred)
        return score

    #Polynomial Linear Regression
    def polyReg(self):
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        poly_reg = PolynomialFeatures(degree = 4)
        X_poly = poly_reg.fit_transform(self.X_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(poly_reg.transform(self.X_test))
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        
        score = self.checkPerformance(self.y_test, y_pred)
        return score

    #Support Vector Regression
    def svReg(self, path, test_size=0.2, random_state=0):
        dataset = pd.read_csv(path)
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values
        y = y.reshape(len(y),1)

        self.splitDataset(X, y, dataset, test_size, random_state)

        from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        sc_y = StandardScaler()
        X_train = sc_X.fit_transform(self.X_train)
        y_train = sc_y.fit_transform(self.y_train)

        # Training the SVR model on the Training set
        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')
        regressor.fit(X_train, y_train.ravel())

        # Predicting the Test set results
        y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(self.X_test)))
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))

        score = self.checkPerformance(self.y_test, y_pred)
        return score

    #Random Forest Regression
    def rfReg(self):
        # Training the Random Forest Regression model on the whole dataset
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor(n_estimators = 12, random_state = 0)
        regressor.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        score = self.checkPerformance(self.y_test, y_pred)
        return score

    #Decision Tree Regression
    def dtReg(self):
        # Training the Decision Tree Regression model on the Training set
        from sklearn.tree import DecisionTreeRegressor
        regressor = DecisionTreeRegressor(random_state = 0)
        regressor.fit(self.X_train, self.y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(self.X_test)
        np.set_printoptions(precision=2)
        # print(np.concatenate((y_pred.reshape(len(y_pred),1), self.y_test.reshape(len(self.y_test),1)),1))
        score = self.checkPerformance(self.y_test, y_pred)
        return score

class Classification:
    def __init__(self, path, test_size=0.2, random_state=0):
        dataset = pd.read_csv(path)
        X = dataset.iloc[:,:-1].values
        y = dataset.iloc[:,-1].values

        self.splitDataset(X, y, dataset, test_size, random_state)
        self.featureScale(self.X_train, self.X_test)
    
    def splitDataset(self, X, y, dataset, test_size, random_state):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def featureScale(self, X_train, X_test):
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        self.X_train = sc.fit_transform(X_train)
        self.X_test = sc.transform(X_test)

    def logisReg(self):
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def knn(self):
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def nbcla(self):
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def dtcla(self):
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def rfcla(self):
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def svm(self):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def ksvm(self):
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(self.X_train, self.y_train)
        return self.results(classifier)

    def results(self, classifier):
        from sklearn.metrics import confusion_matrix, accuracy_score
        y_pred = classifier.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        # print(cm)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

if __name__ == "__main__":
    dataFilePathReg = "F:\MLudemy\MLAZ\\100DaysMLCode\modelSel\RegData.csv"
    dataFilePathCla = "F:\MLudemy\MLAZ\\100DaysMLCode\modelSel\BreastCancerData.csv"
    # reg = Regression(dataFilePathReg)
    cla = Classification(dataFilePathCla)
    # print("Simple Linear Regression Score = ",reg.simpleReg())

    # print("Multiple Regression Score = ",reg.multiReg())
    # print("Polynomial Regression Score = ",reg.polyReg())
    # print("Desicion Tree Regression Score = ",reg.dtReg())
    # print("Random Forest Regression Score = ",reg.rfReg())
    # print("Support Vector Regression Score = ",reg.svReg(dataFilePath))

    
    print("Logistic Regression Score = ", cla.logisReg())
    print("KNN Score = ", cla.knn())
    print("Naive Bayes Score = ", cla.nbcla())
    print("Desicion Tree Classification Score = ", cla.dtcla())
    print("Random Forest Classification Score = ", cla.rfcla())
    print("Support Vector Machine Score = ", cla.svm())
    print("Kernel Support Vector Machine Score = ", cla.ksvm())
    
