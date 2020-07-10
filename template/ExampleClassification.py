from mlTemp import Classification

dataFilePathCla = "F:\MLudemy\MLAZ\MachineLearningPrograms\\template\BreastCancerData.csv"
cla = Classification(dataFilePathCla)
    
print("Logistic Regression Score = ", cla.logisReg())
print("KNN Score = ", cla.knn())
print("Naive Bayes Score = ", cla.nbcla())
print("Desicion Tree Classification Score = ", cla.dtcla())
print("Random Forest Classification Score = ", cla.rfcla())
print("Support Vector Machine Score = ", cla.svm())
print("Kernel Support Vector Machine Score = ", cla.ksvm())