from mlTemp import Regression

dataFilePathReg = "F:\MLudemy\MLAZ\MachineLearningPrograms\\template\RegData.csv"
reg = Regression(dataFilePathReg)

print("Simple Linear Regression Score = ",reg.simpleReg())
print("Multiple Regression Score = ",reg.multiReg())
print("Polynomial Regression Score = ",reg.polyReg())
print("Desicion Tree Regression Score = ",reg.dtReg())
print("Random Forest Regression Score = ",reg.rfReg())
print("Support Vector Regression Score = ",reg.svReg(dataFilePathReg))