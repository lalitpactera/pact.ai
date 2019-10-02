import pandas as pd
class NormalizeColumns:
    def normalizeAllMinMax(pandas_object):
        for col in pandas_object.columns:
                pandas_object[col] = (pandas_object[col] - pandas_object[col].min())/(pandas_object[col].max() - pandas_object[col].min()) 
                
    def normalizeAllStdScl(pandas_object):
        for col in pandas_object.columns:
            pandas_object[col] = pandas_object[col]/pandas_object[col].std()