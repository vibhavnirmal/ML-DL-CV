import pandas as pd
import numpy as np

dataset = pd.read_csv("F:\MLudemy\MLAZ\\100DaysMLCode\Regression\\bank\\bank-data.csv")

null_counts = dataset.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)