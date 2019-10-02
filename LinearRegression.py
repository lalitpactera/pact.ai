import pandas as pd
from sklearn import datasets, linear_model
from sklearn.pipeline import Pipeline
import numpy as np

class LinearRegression:
    def classify(X_train, y_train):
        best_clf_l1 = Pipeline([
            ('std', StandardScaler()),
            ('lr', linear_model.LinearRegression())])
		best_clf_l1.fit(X_train, y_train)
        
        return best_clf
        