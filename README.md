# Machine Learning Programs
[![CodeFactor](https://www.codefactor.io/repository/github/vibhavnirmal/machinelearningprograms/badge)](https://www.codefactor.io/repository/github/vibhavnirmal/machinelearningprograms)

Currently working on various machine learning models and algorithms with different types of datasets available online.

## My Understandings on different models and methods:-

### Pros/Cons of Ensemble learning:

If you have models with high variance (they over-fit your data), then you are likely to benefit from using bagging. 

If you have biased models, it is better to combine them with Boosting. 

If you use the wrong ensemble method for your setting, you are not going to do better. 
For example, using Bagging with a biased model is not going to help.

Also, if you need to work in a probabilistic setting, ensemble methods may not work either. It is known that Boosting (in its most popular forms like AdaBoost) delivers poor probability estimates. 

If you would like to have a model that allows you to reason about your data, not only classification, you might be better off with a graphical model.

## Simple Ensemble Techniques:
### MAX Voting: 
Used for classification, predictions we get from majority of the models counts as final prediction.
(taking mod of all the predictions) (can use voting classifier from sklearn)

