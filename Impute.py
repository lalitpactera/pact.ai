import pandas as pd
class Impute:
    def imputeMean(pandas_object, method):
        if method == 'mean':
            pandas_object.fillna(pandas_object.mean(), inplace=True)
        else:
            pandas_object.fillna(pandas_object.mean(), inplace=True)