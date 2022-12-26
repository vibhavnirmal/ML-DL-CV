# Index

[Ridge regression vs Lasso regression](#Ridge-regression-vs-Lasso-regression)

## Ridge regression vs Lasso regression

In presence of few variables with medium / large sized effect, use lasso regression.
Lasso regression (L1) does both variable selection and parameter shrinkage.

In presence of many variables with small / medium sized effect, use ridge regression.
Ridge regression only does parameter shrinkage and end up including all the coefficients in the model.
In presence of correlated variables, ridge regression might be the preferred choice. Also, ridge regression works best in situations where the least square estimates have higher variance. Therefore, it depends on our model objective.
