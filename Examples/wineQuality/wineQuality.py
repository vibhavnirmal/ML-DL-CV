import pandas as pd
from matplotlib import pyplot as plt

train = pd.read_csv("F:\MLudemy\MLAZ\MachineLearningPrograms\Regression\wineQuality\winequality-red.csv", delimiter=';')

print(train.head())
print(train.info())