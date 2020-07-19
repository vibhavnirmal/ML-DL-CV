# Classification

## My Understandings on different models and methods:-

### Pros/Cons of Ensemble learning:

If you need to work in a probabilistic setting, ensemble methods may not work. It is known that Boosting (in its most popular forms like AdaBoost) delivers poor probability estimates.
(If you would like to have a model that allows you to reason about your data, not only classification, you might be better off with a graphical model.)

## Simple Ensemble Techniques:
### MAX Voting: 
Used for classification, predictions we get from majority of the models counts as final prediction.
(taking mod of all the predictions) (can use voting classifier from sklearn)

### Averaging and Weighted Avarage:
Mostly used for regression and calculating probabities for classificaion models

## Advanced Ensemble Techniques:
### 1. Stacking:
Uses predictions from different models to build a new model. The new model is used to get the predictions on test set 
[MoreHere](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)

### 2. Blending:
Uses validation(holdout) set from train set to make predictions. Predictions are made only on holdout set.
The validation set and predictions are used to build a new model and then tested on test set.

### Difference between Stacking and Blending:
The difference between stacking and blending is that Stacking uses out-of-fold predictions for the train set of the next layer (i.e meta-model), and Blending uses a validation set (let’s say, 10-15% of the training set) to train the next layer.

### 3. Bagging(bootstrap aggregating):
Combines results of different models to get a generalized result.
If you have models with high variance (they over-fit your data), then you are likely to benefit from using bagging. 
Using Bagging with a biased model is not going to help.

#### Bootstrapping:
It uses bags to get fair idea of the complete dataset
The size of subsets created for bagging may be less than the original set.
From original dataset, subsets(bags) of observations are created with replacement
The size of subset is same as the size of original dataset.
A base(weak) model is created on each subset
Final predictions are determined by combining predictions from all weak models

#### MODELS:
* Bagging meta-estimator ([BaggingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html))
* [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

### 4. Boosting:
It is a sequential process. Where each subsequent model tries to correct errors of the previous model.
So, the succeeding models are dependent on previous models.
It combines many weak learners to form a strong learner.
The individual models dont work best of whole dataset, but they may work better on some part of the dataset. Thus each model boosts the performance of the ensemble.
If you have biased models, it is better to combine them with Boosting.

#### MODELS:
* [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
* XGBoost ([ExtremeGradientBootsing](https://xgboost.readthedocs.io/en/latest/))
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/) (It beats all other models when dataset is large..)
* [CatBoost](https://catboost.ai/docs/concepts/python-reference_parameters-list.html) (No need to perform one hot encoding)